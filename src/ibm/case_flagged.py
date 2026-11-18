"""The real setting: group the transaction model's flagged payments into cases and name each case.

The flagged payments are the only thing the classifier sees, so a case may miss payments of its
pattern and may include flagged payments from unrelated patterns. The classifier is the same one
trained on whole attempts (case_typology.py).

Example:
    cases = build_flagged_cases(frame, flagged_rows, classifier)
    result = case_level_result(cases)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.ibm.case_groups import link_flagged_payments, rows_from_payments
from src.ibm.case_typology import accuracy_of, predict_shapes, wilson_interval
from src.ibm.loader import NOT_LAUNDERING, UNLISTED_TYPOLOGY
from src.ibm.sft_format import TYPOLOGIES


@dataclass(frozen=True)
class FlaggedCase:
    """One group of flagged payments: where they are in the table, what they really are, what we named it."""

    positions: List[int]
    typologies: List[str]
    predicted: str

    @property
    def truth(self) -> Optional[str]:
        """The most common listed typology among its payments, or None if it has no listed laundering payment."""
        listed = [t for t in self.typologies if t in TYPOLOGIES]
        return Counter(listed).most_common(1)[0][0] if listed else None

    @property
    def is_pure(self) -> bool:
        """Every payment in the case belongs to the same listed typology."""
        return len(set(self.typologies)) == 1 and self.typologies[0] in TYPOLOGIES


def flagged_by_threshold(scores: np.ndarray, period: slice, threshold: float) -> np.ndarray:
    """Row positions in the period whose score reaches the threshold chosen on validation."""
    start = period.start or 0
    return np.flatnonzero(scores[period] >= threshold) + start


def flagged_by_budget(scores: np.ndarray, period: slice, count: int) -> np.ndarray:
    """Row positions of the `count` highest-scored payments in the period."""
    start = period.start or 0
    return np.argsort(-scores[period], kind="stable")[:count] + start


def build_flagged_cases(frame: pd.DataFrame, positions: np.ndarray, classifier) -> List[FlaggedCase]:
    """Link the flagged payments into cases and name each one."""
    part = frame.iloc[positions]
    groups = link_flagged_payments(part)
    rows = [rows_from_payments(part.iloc[group]) for group in groups]
    names = predict_shapes(classifier, rows)
    return [
        FlaggedCase([int(positions[i]) for i in group], [str(part["typology"].iloc[i]) for i in group], name)
        for group, name in zip(groups, names)
    ]


def case_level_result(cases: List[FlaggedCase]) -> Dict[str, object]:
    """Case counts and accuracy on cases that contain at least one listed laundering payment."""
    with_truth = [c for c in cases if c.truth is not None]
    pure = [c for c in with_truth if c.is_pure]
    right = sum(1 for c in with_truth if c.predicted == c.truth)
    return {
        "cases": len(cases),
        "cases_with_listed_laundering": len(with_truth),
        "cases_without": len(cases) - len(with_truth),
        "pure_cases": len(pure),
        "accuracy": accuracy_of([c.truth for c in with_truth], [c.predicted for c in with_truth]),
        "accuracy_range": wilson_interval(right, len(with_truth)),
        "pure_accuracy": accuracy_of([c.truth for c in pure], [c.predicted for c in pure]),
        "median_case_size": float(np.median([len(c.positions) for c in cases])) if cases else 0.0,
        "by_size": accuracy_by_case_size(cases),
    }


SIZE_BUCKETS = (("1 payment", 1, 1), ("2 to 4 payments", 2, 4), ("5 or more payments", 5, 10**9))


def accuracy_by_case_size(cases: List[FlaggedCase]) -> Dict[str, Dict[str, float]]:
    """Case-level accuracy for single-payment cases, small cases and larger cases.

    Shows where the real setting breaks: a shape cannot be read from one or two payments."""
    result = {}
    for name, low, high in SIZE_BUCKETS:
        chosen = [c for c in cases if c.truth is not None and low <= len(c.positions) <= high]
        result[name] = {
            "cases": len(chosen),
            "accuracy": accuracy_of([c.truth for c in chosen], [c.predicted for c in chosen]),
        }
    return result


def payment_labels(cases: List[FlaggedCase]) -> Dict[int, str]:
    """Each flagged payment inherits the name of its case."""
    return {position: case.predicted for case in cases for position in case.positions}


def payment_accuracy(frame: pd.DataFrame, rows: List[int], inherited: Dict[int, str]) -> Dict[str, float]:
    """Accuracy over listed-laundering payments: unflagged ones count as wrong, and flagged-only is shown too."""
    listed = [r for r in rows if frame["typology"].iat[r] in TYPOLOGIES]
    flagged = [r for r in listed if r in inherited]
    right = sum(1 for r in flagged if inherited[r] == frame["typology"].iat[r])
    return {
        "payments": len(listed),
        "flagged": len(flagged),
        "accuracy_counting_missed_as_wrong": right / len(listed) if listed else 0.0,
        "accuracy_among_flagged": right / len(flagged) if flagged else 0.0,
    }


def listed_rows_in(frame: pd.DataFrame, period: slice) -> List[int]:
    """Row positions of listed-laundering payments in a period."""
    part = frame["typology"].iloc[period]
    keep = ~part.isin([NOT_LAUNDERING, UNLISTED_TYPOLOGY])
    return (np.flatnonzero(keep.to_numpy()) + (period.start or 0)).tolist()
