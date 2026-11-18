"""Train and score a LightGBM model on the IBM transactions, with a temporal split.

The first 60% of the timeline trains the model, the next 20% picks the alert
threshold and stops training early, and the last 20% is the test. Time order
matters: a model that saw next week's transactions would look better than it is.
Laundering is rare, so the rare class gets a larger weight (scale_pos_weight).

Example:
    python -m src.ibm.train
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.ibm.features import FEATURE_COLUMNS, build_features
from src.ibm.loader import add_account_codes, load_labelled_transactions

TRAIN_SHARE = 0.60
VALIDATION_SHARE = 0.20  # the remaining 20% is the test period
RANDOM_SEED = 42
PRECISION_AT_K = 100  # an analyst can look at about a hundred alerts a day
MAX_TREES = 400
EARLY_STOPPING_ROUNDS = 30  # stop when validation PR-AUC has not improved for this many trees
LEARNING_RATE = 0.05
LEAVES_PER_TREE = 7  # chosen on validation PR-AUC: 7 beat 15 and 63 (bigger trees overfit)
# Class weights tried for the rare class. "balanced" means negatives / positives (about 1,300).
# Chosen on validation PR-AUC only; the full balanced weight ruined the ranking (see the report).
CANDIDATE_WEIGHTS = (1.0, 10.0, "balanced")


@dataclass
class ModelScores:
    """How one set of scores did on one period."""

    minority_f1: float
    precision: float
    recall: float
    pr_auc: float
    precision_at_100: float
    alerts: int


def temporal_split(row_count: int) -> Tuple[slice, slice, slice]:
    """Row ranges for train, validation and test, in time order (rows are already sorted by time)."""
    train_end = int(row_count * TRAIN_SHARE)
    validation_end = int(row_count * (TRAIN_SHARE + VALIDATION_SHARE))
    return slice(0, train_end), slice(train_end, validation_end), slice(validation_end, row_count)


def precision_at_top(truth: np.ndarray, scores: np.ndarray, top_count: int = PRECISION_AT_K) -> float:
    """Share of the `top_count` highest-scored transactions that really are laundering."""
    best = np.argsort(-scores, kind="stable")[:top_count]
    return float(truth[best].mean()) if len(best) else 0.0


def best_f1_threshold(truth: np.ndarray, scores: np.ndarray) -> float:
    """The score cut-off that gives the best minority-class F1 (chosen on validation only)."""
    precision, recall, thresholds = precision_recall_curve(truth, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def score_period(truth: np.ndarray, scores: np.ndarray, threshold: float) -> ModelScores:
    """Minority-class F1, precision, recall, PR-AUC and precision@100 at a fixed threshold."""
    flagged = scores >= threshold
    true_positive = int((flagged & (truth == 1)).sum())
    precision = true_positive / flagged.sum() if flagged.any() else 0.0
    recall = true_positive / max(int((truth == 1).sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ModelScores(
        f1,
        precision,
        recall,
        float(average_precision_score(truth, scores)),
        precision_at_top(truth, scores),
        int(flagged.sum()),
    )


def resolve_weight(weight, truth: np.ndarray, train: slice):
    """Turn "balanced" into negatives / positives for the training rows; numbers pass through."""
    if weight != "balanced":
        return float(weight)
    positives = max(int(truth[train].sum()), 1)
    return (len(truth[train]) - positives) / positives


def fit_model(features: pd.DataFrame, truth: np.ndarray, train: slice, validation: slice, weight=1.0):
    """Train LightGBM with the rare class weighted by `weight`; stop early on validation PR-AUC."""
    import lightgbm

    weight = resolve_weight(weight, truth, train)
    model = lightgbm.LGBMClassifier(
        n_estimators=MAX_TREES,
        learning_rate=LEARNING_RATE,
        num_leaves=LEAVES_PER_TREE,
        scale_pos_weight=weight,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(
        features.iloc[train],
        truth[train],
        eval_set=[(features.iloc[validation], truth[validation])],
        eval_metric="average_precision",
        callbacks=[lightgbm.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the labelled transactions and compute their features. Returns (transactions, features)."""
    transactions = add_account_codes(load_labelled_transactions())
    return transactions, build_features(transactions)


def train_and_score(transactions: pd.DataFrame, features: pd.DataFrame) -> Dict[str, object]:
    """Try each class weight, keep the one with the best validation PR-AUC, score validation and test."""
    truth = transactions["is_laundering"].to_numpy()
    train, validation, test = temporal_split(len(transactions))
    tried: Dict[str, float] = {}
    best = None
    for weight in CANDIDATE_WEIGHTS:
        model = fit_model(features, truth, train, validation, weight)
        scores = model.predict_proba(features)[:, 1]
        validation_pr_auc = float(average_precision_score(truth[validation], scores[validation]))
        tried[str(weight)] = validation_pr_auc
        if best is None or validation_pr_auc > best[0]:
            best = (validation_pr_auc, weight, model, scores)
    _, weight, model, scores = best
    threshold = best_f1_threshold(truth[validation], scores[validation])
    return {
        "model": model,
        "scores": scores,
        "threshold": threshold,
        "weight": weight,
        "validation_pr_auc_by_weight": tried,
        "validation": score_period(truth[validation], scores[validation], threshold),
        "test": score_period(truth[test], scores[test], threshold),
    }


def top_feature_names(model, count: int = 8) -> List[Tuple[str, int]]:
    """The features the trees used most, as (name, split count) pairs."""
    pairs = sorted(zip(FEATURE_COLUMNS, model.booster_.feature_importance("split")), key=lambda p: -p[1])
    return [(name, int(value)) for name, value in pairs[:count]]


def gain_importance(model) -> List[Tuple[str, float]]:
    """Each feature's total gain (how much it improved the trees' fit), largest first."""
    gains = model.booster_.feature_importance("gain")
    return sorted(zip(model.booster_.feature_name(), [float(g) for g in gains]), key=lambda pair: -pair[1])


def train_without_features(
    transactions: pd.DataFrame, features: pd.DataFrame, weight, dropped: List[str]
) -> Dict[str, object]:
    """Ablation: retrain with one fixed class weight and some feature columns removed, then score test."""
    truth = transactions["is_laundering"].to_numpy()
    train, validation, test = temporal_split(len(transactions))
    kept = features.drop(columns=dropped)
    model = fit_model(kept, truth, train, validation, weight)
    scores = model.predict_proba(kept)[:, 1]
    threshold = best_f1_threshold(truth[validation], scores[validation])
    return {
        "model": model,
        "scores": scores,
        "threshold": threshold,
        "test": score_period(truth[test], scores[test], threshold),
    }


if __name__ == "__main__":
    all_transactions, all_features = prepare_data()
    result = train_and_score(all_transactions, all_features)
    print("threshold", result["threshold"])
    print("validation", result["validation"])
    print("test", result["test"])
