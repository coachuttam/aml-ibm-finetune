"""Extra report sections: feature gain, the payment-format ablation, and the investigation summary."""

from __future__ import annotations

from typing import Dict, List

from src.ibm.case_text import typology_words
from src.ibm.train import ModelScores, gain_importance

ABLATION_NAME = "LightGBM − payment format"


def markdown_table(header: List[str], rows: List[List[str]]) -> str:
    """A markdown table from a header and rows of text."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def gain_table_rows(model) -> List[List[str]]:
    """Feature, total gain and share of all gain, largest first."""
    pairs = gain_importance(model)
    total = sum(gain for _, gain in pairs) or 1.0
    return [[name, "%.0f" % gain, "%.1f%%" % (100 * gain / total)] for name, gain in pairs]


def ablation_row(scores: ModelScores) -> List[str]:
    """The ablation's test result in the Overall table's columns."""
    return [
        ABLATION_NAME,
        str(scores.alerts),
        "%.4f" % scores.precision,
        "%.3f" % scores.recall,
        "%.3f" % scores.minority_f1,
        "%.3f" % scores.pr_auc,
        "%.2f" % scores.precision_at_100,
    ]


def accuracy_text(score: Dict[str, float]) -> str:
    return "%d of %d (%.0f%%)" % (score["correct"], score["rows"], 100 * score["accuracy"])


def investigation_lines(investigation: Dict[str, object]) -> List[str]:
    """Plain-English summary of the top-alert investigations and the typology scores (old versus new)."""
    records = investigation["records"]
    guesses = sorted({typology_words(record["typology"]) for record in records})
    config = investigation["graph_config"]
    return [
        "- Investigated the %d highest-scored test alerts; %d of them are truly laundering."
        % (len(records), investigation["top_really_laundering"]),
        "- Typology accuracy against the patterns file, top 20 alerts: graph checks + fallback %s; "
        "classifier only %s."
        % (accuracy_text(investigation["top_accuracy"]), accuracy_text(investigation["classifier_top_accuracy"])),
        "- Typology accuracy, every test-period laundering row with a listed typology: graph checks + fallback %s; "
        "classifier only %s."
        % (accuracy_text(investigation["test_accuracy"]), accuracy_text(investigation["classifier_test_accuracy"])),
        "- Graph checks decided %.0f%% of the top 20 and %.0f%% of all test rows; the classifier filled the rest. "
        "Settings chosen on the training period: fan degree k = %d, cycles up to %d payments."
        % (
            100 * investigation["top_graph_share"],
            100 * investigation["test_graph_share"],
            config["degree_min"],
            config["max_cycle_hops"],
        ),
        "- Typologies suggested for the top 20: %s." % ", ".join(guesses),
        "- Graph checks look at the 7-day payment neighbourhood (cycle, scatter-gather, bipartite, fan-in/out, "
        "gather-scatter). The fallback is a LightGBM classifier trained on training-period laundering rows only. "
        "Each investigation is saved in `results/ibm_investigations.jsonl`; nothing is filed or sent.",
    ]


def budget_section(rows: List[List[str]], needed: int, rules_recall: float, chart_name: str) -> List[str]:
    """Markdown for the alert-budget table, the match-the-rules number and the chart link."""
    table = markdown_table(["Alert budget", "Recall", "Precision"], rows)
    match = "The model needs **%s alerts** to match the rules' recall of %.1f%%." % (
        "{:,}".format(needed),
        100 * rules_recall,
    )
    return ["## Alert budget (test period)", table, match, "![Recall versus alerts](%s)" % chart_name]
