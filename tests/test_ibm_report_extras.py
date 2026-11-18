from __future__ import annotations

from src.ibm.report_extras import ABLATION_NAME, ablation_row, investigation_lines
from src.ibm.train import ModelScores


def test_ablation_row_has_the_name_and_the_overall_table_columns():
    row = ablation_row(ModelScores(0.2, 0.3, 0.4, 0.5, 0.6, 7))
    assert row == [ABLATION_NAME, "7", "0.3000", "0.400", "0.200", "0.500", "0.60"]


def test_investigation_summary_reports_old_and_new_accuracy():
    investigation = {
        "records": [{"typology": "FAN_OUT"}],
        "top_really_laundering": 1,
        "top_accuracy": {"rows": 1, "correct": 1, "accuracy": 1.0},
        "classifier_top_accuracy": {"rows": 1, "correct": 0, "accuracy": 0.0},
        "test_accuracy": {"rows": 10, "correct": 4, "accuracy": 0.4},
        "classifier_test_accuracy": {"rows": 10, "correct": 3, "accuracy": 0.3},
        "top_graph_share": 1.0,
        "test_graph_share": 0.5,
        "graph_config": {"degree_min": 8, "max_cycle_hops": 5},
    }
    text = "\n".join(investigation_lines(investigation))
    assert "1 of 1 (100%)" in text and "0 of 1 (0%)" in text
    assert "4 of 10 (40%)" in text and "3 of 10 (30%)" in text
    assert "k = 8" in text and "Fan-out" in text
