"""Alert budget: how much laundering the model finds if an analyst can only work N alerts.

The rules raise a fixed number of alerts. The model ranks every transaction, so we
can ask what recall it reaches at the same budget, and how many alerts it needs to
match the rules' recall.

Example:
    rows = budget_rows(truth, scores, [10_000, 25_000])
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

CHART_PATH = Path("results/ibm_recall_vs_alerts.png")
CURVE_POINTS = 400  # enough points for a smooth line without a huge image file


def hits_in_top(truth: np.ndarray, scores: np.ndarray, count: int) -> int:
    """How many true laundering rows are among the `count` highest scores."""
    best = np.argsort(-scores, kind="stable")[:count]
    return int(truth[best].sum())


def recall_at(truth: np.ndarray, scores: np.ndarray, count: int) -> float:
    """Share of all laundering rows found within the top `count` alerts."""
    positives = int(truth.sum())
    return hits_in_top(truth, scores, count) / positives if positives else 0.0


def alerts_needed_for_recall(truth: np.ndarray, scores: np.ndarray, target_recall: float) -> int:
    """The fewest top-ranked alerts whose recall reaches `target_recall` (0 if the target is 0)."""
    found = np.cumsum(truth[np.argsort(-scores, kind="stable")])
    needed = target_recall * int(truth.sum())
    reached = np.flatnonzero(found >= needed - 1e-9)
    return int(reached[0]) + 1 if len(reached) else len(truth)


def budget_rows(truth: np.ndarray, scores: np.ndarray, budgets: Sequence[int]) -> List[List[str]]:
    """Table rows: alerts, recall and precision at each budget."""
    rows = []
    for budget in budgets:
        hits = hits_in_top(truth, scores, budget)
        recall = hits / max(int(truth.sum()), 1)
        rows.append(["{:,}".format(budget), "%.3f" % recall, "%.4f" % (hits / budget)])
    return rows


def curve(truth: np.ndarray, scores: np.ndarray, max_alerts: int) -> np.ndarray:
    """Recall at log-spaced budgets up to `max_alerts`, as (alerts, recall) rows.

    Even spacing would draw a straight line at the low end of a log axis.
    """
    found = np.cumsum(truth[np.argsort(-scores, kind="stable")])
    max_alerts = min(max_alerts, len(truth))
    counts = np.unique(np.geomspace(1, max_alerts, CURVE_POINTS).astype(int))
    return np.column_stack([counts, found[counts - 1] / max(int(truth.sum()), 1)])


def draw_chart(truth: np.ndarray, scores: np.ndarray, rules_point: Sequence[float], path: Path = CHART_PATH) -> None:
    """Save recall versus alerts as a PNG, with the rules as one point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = curve(truth, scores, int(max(rules_point[0] * 1.1, 1000)))
    figure, axis = plt.subplots(figsize=(7, 4.2), dpi=150)
    axis.plot(points[:, 0], points[:, 1], color="#1f5fa8", linewidth=2, label="LightGBM (ranked)")
    axis.scatter([rules_point[0]], [rules_point[1]], color="#c2410c", s=60, zorder=3, label="Rules R1-R5 (one point)")
    axis.set_xscale("log")
    axis.set_xlabel("Alerts raised (test period, log scale)")
    axis.set_ylabel("Recall (share of laundering found)")
    axis.set_title("Recall versus alert budget")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
