"""Run the case-level typology experiment and write results/case_typology.md.

    python -m src.ibm.case_report

Step 1 (ceiling): name whole laundering attempts from their shape.
Step 2 (real setting): name cases built only from the transaction model's flagged payments.
Only training-period attempts are learned from. Nothing is tuned on the test period.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.ibm.build_sft import load_examples
from src.ibm.case_flagged import (
    build_flagged_cases,
    case_level_result,
    flagged_by_budget,
    flagged_by_threshold,
    listed_rows_in,
    payment_accuracy,
    payment_labels,
)
from src.ibm.case_groups import make_attempts, split_attempts, split_boundaries
from src.ibm.case_report_text import write_markdown
from src.ibm.case_typology import evaluate_cases, predict_shapes, train_shape_classifier
from src.ibm.loader import parse_patterns
from src.ibm.train import prepare_data, temporal_split, train_and_score

REPORT_PATH = Path("results/case_typology.md")
EXTRA_BUDGET = 5000  # a second, larger alert budget, to show how the answer moves with the number of flags


def oracle_results(attempts: list, classifier) -> Dict[str, object]:
    """Typology scores of the classifier on whole attempts of one split."""
    truths = [a.typology for a in attempts]
    return evaluate_cases(truths, predict_shapes(classifier, [a.rows for a in attempts]))


def flagged_results(frame, scores, classifier, positions, sft_rows: List[int], test_period) -> Dict[str, object]:
    """Case-level and payment-level scores for one set of flagged payments."""
    cases = build_flagged_cases(frame, positions, classifier)
    inherited = payment_labels(cases)
    return {
        "flagged_payments": len(positions),
        "case_level": case_level_result(cases),
        "all_listed_test_payments": payment_accuracy(frame, listed_rows_in(frame, test_period), inherited),
        "sft_test_payments": payment_accuracy(frame, sft_rows, inherited),
    }


def main() -> None:
    """Build everything and write the report."""
    frame, features = prepare_data()
    model_result = train_and_score(frame, features)
    scores, threshold = model_result["scores"], model_result["threshold"]
    disjoint, time_only = split_attempts(make_attempts(parse_patterns()), split_boundaries(frame))
    classifier = train_shape_classifier(disjoint["train"])
    _, _, test_period = temporal_split(len(frame))
    sft_rows = [int(e["alert_id"].split("-")[1]) for e in load_examples("test") if e["typology"] != "NONE"]
    results = {
        "attempt_counts": {
            "disjoint": {k: len(v) for k, v in disjoint.items()},
            "time_only": {k: len(v) for k, v in time_only.items()},
        },
        "oracle_val": oracle_results(disjoint["val"], classifier),
        "oracle_test": oracle_results(disjoint["test"], classifier),
        "oracle_test_time_only": oracle_results(time_only["test"], classifier),
        "importance": sorted(
            zip(classifier.booster_.feature_name(), classifier.booster_.feature_importance("gain")), key=lambda p: -p[1]
        ),
        "threshold_flags": flagged_results(
            frame, scores, classifier, flagged_by_threshold(scores, test_period, threshold), sft_rows, test_period
        ),
        "budget_flags": flagged_results(
            frame, scores, classifier, flagged_by_budget(scores, test_period, EXTRA_BUDGET), sft_rows, test_period
        ),
    }
    REPORT_PATH.write_text(write_markdown(results), encoding="utf-8")
    print(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
