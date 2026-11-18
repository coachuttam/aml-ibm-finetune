"""Run the rules and the model on the same periods and write results/ibm_compare.md.

Example:
    python -m src.ibm.compare
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.ibm.budget import CHART_PATH, alerts_needed_for_recall, budget_rows, draw_chart
from src.ibm.hybrid import best_rule_for_typology, hybrid_flags
from src.ibm.investigate import run_investigations, save_records
from src.ibm.report_extras import (
    ablation_row,
    budget_section,
    gain_table_rows,
    investigation_lines,
    markdown_table,
)
from src.ibm.rules import RULES, run_rules, score_flags
from src.ibm.train import (
    ModelScores,
    prepare_data,
    temporal_split,
    top_feature_names,
    train_and_score,
    train_without_features,
)

REPORT_PATH = Path("results/ibm_compare.md")
NOT_LAUNDERING = "NONE"


def rule_scores_for(frame: pd.DataFrame, flags: pd.DataFrame, period: slice) -> List[List[str]]:
    """Table rows: one per rule plus 'any rule', each with alerts, precision and recall."""
    rows = []
    truth = frame["is_laundering"].iloc[period]
    for column in [rule.rule_id for rule in RULES] + ["any_rule"]:
        score = score_flags(flags[column].iloc[period], truth)
        rows.append([column, str(score["alerts"]), "%.4f" % score["precision"], "%.3f" % score["recall"]])
    return rows


def recall_by_typology(frame: pd.DataFrame, flagged: pd.Series, period: slice) -> Dict[str, float]:
    """For each typology, the share of its transactions that were flagged."""
    part = frame.iloc[period]
    result = {}
    for typology in sorted(set(part["typology"]) - {NOT_LAUNDERING}):
        mask = part["typology"] == typology
        result[typology] = float(flagged.iloc[period][mask].mean())
    return result


def model_row(name: str, scores: ModelScores) -> List[str]:
    return [
        name,
        str(scores.alerts),
        "%.4f" % scores.precision,
        "%.3f" % scores.recall,
        "%.3f" % scores.minority_f1,
        "%.3f" % scores.pr_auc,
        "%.2f" % scores.precision_at_100,
    ]


def flag_row(name: str, frame: pd.DataFrame, flagged: pd.Series, period: slice, pr_auc: str, at_100: str) -> List[str]:
    """A table row (alerts, precision, recall, F1) for any yes/no column."""
    score = score_flags(flagged.iloc[period], frame["is_laundering"].iloc[period])
    precision, recall = score["precision"], score["recall"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return [name, str(score["alerts"]), "%.4f" % precision, "%.3f" % recall, "%.3f" % f1, pr_auc, at_100]


def best_rule_row(frame: pd.DataFrame, flags: pd.DataFrame, period: slice) -> List[str]:
    """The rules' 'any rule' result in the same columns as the model row."""
    truth = frame["is_laundering"].iloc[period]
    score = score_flags(flags["any_rule"].iloc[period], truth)
    precision, recall = score["precision"], score["recall"]
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return [
        "Rules (any of R1-R5)",
        str(score["alerts"]),
        "%.4f" % precision,
        "%.3f" % recall,
        "%.3f" % f1,
        "n/a",
        "n/a (no ranking)",
    ]


def budget_sections(frame: pd.DataFrame, flags: pd.DataFrame, result: Dict[str, object], test: slice) -> List[str]:
    """Recall at several alert budgets, the budget that matches the rules, and the chart (saved as a PNG)."""
    truth = frame["is_laundering"].to_numpy()[test]
    scores = result["scores"][test]
    rules = score_flags(flags["any_rule"].iloc[test], frame["is_laundering"].iloc[test])
    budgets = [rules["alerts"], 50_000, 25_000, 10_000, result["test"].alerts]
    needed = alerts_needed_for_recall(truth, scores, rules["recall"])
    draw_chart(truth, scores, [rules["alerts"], rules["recall"]], CHART_PATH)
    rows = budget_rows(truth, scores, budgets)
    return budget_section(rows, needed, rules["recall"], CHART_PATH.name)


def build_report(
    frame: pd.DataFrame,
    flags: pd.DataFrame,
    result: Dict[str, object],
    ablation: Optional[Dict[str, object]] = None,
    investigation: Optional[Dict[str, object]] = None,
) -> str:
    """The whole markdown report as text."""
    _, _, test = temporal_split(len(frame))
    test_part = frame.iloc[test]
    header = ["System", "Alerts", "Precision", "Recall", "Minority F1", "PR-AUC", "Precision@100"]
    scores = result["scores"]
    model_flags = pd.Series(scores >= result["threshold"], index=frame.index)
    train, _, _ = temporal_split(len(frame))
    chosen_rule = best_rule_for_typology(frame, flags, train, "UNLISTED")
    hybrid = hybrid_flags(model_flags, flags[chosen_rule])
    hybrid_row = flag_row("Hybrid (model OR %s)" % chosen_rule, frame, hybrid, test, "n/a", "n/a")
    rule_recall = recall_by_typology(frame, flags["any_rule"], test)
    model_recall = recall_by_typology(frame, model_flags, test)
    hybrid_recall = recall_by_typology(frame, hybrid, test)
    typology_rows = [
        [name, "%.3f" % rule_recall[name], "%.3f" % model_recall[name], "%.3f" % hybrid_recall[name]]
        for name in rule_recall
    ]
    top_features = ", ".join("%s (%d)" % pair for pair in top_feature_names(result["model"]))
    overall_rows = [best_rule_row(frame, flags, test), model_row("LightGBM", result["test"]), hybrid_row]
    if ablation is not None:
        overall_rows.append(ablation_row(ablation["test"]))
    sections = [
        "## Feature importance (total gain, full model)",
        markdown_table(["Feature", "Gain", "Share"], gain_table_rows(result["model"])),
    ]
    sections += budget_sections(frame, flags, result, test)
    if investigation is not None:
        sections += ["## Investigations of the top 20 test alerts", "\n".join(investigation_lines(investigation))]
    return (
        "\n\n".join(
            [
                "# IBM AML: rules versus LightGBM",
                "Data: IBM HI-Small (%d transactions, %d laundering, base rate %.4f%% in the test period). "
                "Temporal split 60/20/20 by time; the threshold is chosen on validation, "
                "and every number below is the **test** period."
                % (len(frame), int(frame["is_laundering"].sum()), 100 * test_part["is_laundering"].mean()),
                "## Overall (test period)",
                markdown_table(header, overall_rows),
                "## Each rule alone (test period)",
                markdown_table(["Rule", "Alerts", "Precision", "Recall"], rule_scores_for(frame, flags, test)),
                "## Recall by typology (test period)",
                markdown_table(["Typology", "Rules", "LightGBM", "Hybrid"], typology_rows),
                "The hybrid adds %s, the rule with the best F1 on UNLISTED transactions "
                "in the training period (never chosen on test)." % chosen_rule,
                "## Class weight (chosen on validation PR-AUC, never on test)",
                markdown_table(
                    ["Weight on the laundering class", "Validation PR-AUC"],
                    [[name, "%.4f" % value] for name, value in result["validation_pr_auc_by_weight"].items()],
                )
                + "\n\nChosen: %s." % result["weight"],
                "Most-used model features (split counts): %s." % top_features,
                *sections,
                "## Notes",
                "- Rule thresholds were set by looking at the first 60% of the timeline only.\n"
                "- Amounts are in each row's own currency; there are no exchange rates, so amount features are rough.\n"
                "- Features use only past transactions. Account degree and in/out ratio are running totals,\n"
                "  not whole-dataset values.\n"
                "- Rules only say yes or no, so PR-AUC and precision@100 do not apply to them.",
            ]
        )
        + "\n"
    )


def main() -> None:
    frame, features = prepare_data()
    flags = run_rules(frame)
    result = train_and_score(frame, features)
    ablation = train_without_features(frame, features, result["weight"], ["payment_format_code"])
    investigation = run_investigations(frame, result["scores"])
    save_records(investigation["records"])
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(build_report(frame, flags, result, ablation, investigation))
    print("wrote", REPORT_PATH)


if __name__ == "__main__":
    main()
