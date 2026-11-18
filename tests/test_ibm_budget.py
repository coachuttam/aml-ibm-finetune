from __future__ import annotations

import numpy as np

from src.ibm.budget import alerts_needed_for_recall, budget_rows, curve, draw_chart, hits_in_top, recall_at

TRUTH = np.array([0, 1, 0, 1, 0, 0, 1, 0])
SCORES = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4, 0.5, 0.0])


def test_hits_in_top_counts_true_rows_among_the_highest_scores():
    assert hits_in_top(TRUTH, SCORES, 2) == 2
    assert hits_in_top(TRUTH, SCORES, 4) == 3


def test_recall_at_is_the_share_of_all_laundering_found():
    assert recall_at(TRUTH, SCORES, 2) == 2 / 3
    assert recall_at(TRUTH, SCORES, 8) == 1.0


def test_alerts_needed_for_recall_is_the_smallest_budget_that_reaches_it():
    assert alerts_needed_for_recall(TRUTH, SCORES, 2 / 3) == 2
    assert alerts_needed_for_recall(TRUTH, SCORES, 1.0) == 3


def test_budget_rows_report_recall_and_precision_with_thousands_separators():
    rows = budget_rows(TRUTH, SCORES, [2])
    assert rows == [["2", "0.667", "1.0000"]]


def test_curve_never_decreases_as_the_budget_grows():
    points = curve(TRUTH, SCORES, 8)
    assert all(later >= earlier for earlier, later in zip(points[:-1, 1], points[1:, 1]))


def test_draw_chart_writes_a_png_file(tmp_path):
    path = tmp_path / "chart.png"
    draw_chart(TRUTH, SCORES, [6, 0.5], path)
    assert path.read_bytes()[:4] == b"\x89PNG"
