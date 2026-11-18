from __future__ import annotations

import numpy as np
import pandas as pd

from src.ibm.features import FEATURE_COLUMNS
from src.ibm.train import (
    best_f1_threshold,
    fit_model,
    precision_at_top,
    resolve_weight,
    score_period,
    temporal_split,
)


def test_temporal_split_is_sixty_twenty_twenty_in_time_order():
    train, validation, test = temporal_split(100)
    assert (train.start, train.stop) == (0, 60)
    assert (validation.start, validation.stop) == (60, 80)
    assert (test.start, test.stop) == (80, 100)


def test_precision_at_top_counts_hits_among_the_highest_scores():
    truth = np.array([1, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert precision_at_top(truth, scores, top_count=2) == 0.5


def test_score_period_gives_perfect_scores_when_scores_match_truth():
    truth = np.array([1, 0, 1, 0])
    result = score_period(truth, np.array([0.9, 0.1, 0.8, 0.2]), threshold=0.5)
    assert (result.minority_f1, result.precision, result.recall, result.pr_auc) == (1.0, 1.0, 1.0, 1.0)


def test_score_period_handles_a_threshold_that_flags_nothing():
    result = score_period(np.array([1, 0]), np.array([0.2, 0.1]), threshold=0.9)
    assert result.minority_f1 == 0.0 and result.alerts == 0


def test_best_threshold_separates_two_clean_groups():
    truth = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert 0.3 < best_f1_threshold(truth, scores) <= 0.8


def test_fit_model_learns_a_simple_signal_with_class_weights():
    generator = np.random.RandomState(0)
    signal = generator.rand(600)
    features = pd.DataFrame({name: generator.rand(600) for name in FEATURE_COLUMNS})
    features["log_amount_paid"] = signal
    truth = (signal > 0.9).astype(int)
    model = fit_model(features, truth, slice(0, 400), slice(400, 600))
    scores = model.predict_proba(features)[:, 1]
    assert scores[truth == 1].mean() > scores[truth == 0].mean()


def test_balanced_weight_is_negatives_divided_by_positives():
    truth = np.array([0] * 90 + [1] * 10)
    assert resolve_weight("balanced", truth, slice(0, 100)) == 9.0
    assert resolve_weight(5, truth, slice(0, 100)) == 5.0
