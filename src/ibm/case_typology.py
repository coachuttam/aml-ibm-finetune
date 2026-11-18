"""Name a case's typology from its shape, and score the result.

A small LightGBM learns "shape numbers -> typology" from the training-period attempts only (shape
numbers come from case_shape.py). It never sees account names, amounts' meaning or any label other
than the typology it is trained to name. Python counts the shape; the classifier only labels.

Example:
    classifier = train_shape_classifier(train_attempts)
    guesses = predict_shapes(classifier, [attempt.rows for attempt in test_attempts])
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import lightgbm
import numpy as np
import pandas as pd

from src.ibm.case_groups import Attempt
from src.ibm.case_shape import SHAPE_FEATURES, CaseRows, shape_features
from src.ibm.sft_format import TYPOLOGIES
from src.metrics import score_label

RANDOM_SEED = 42  # the project's seed
TREES = 200  # about 200 training cases; more trees only memorise them
LEAVES = 5  # small trees for a small training set
LEARNING_RATE = 0.05
MIN_CASES_PER_LEAF = 5  # with 8 classes and ~200 cases, a leaf needs a few cases to mean anything


def shape_table(cases: List[CaseRows]) -> pd.DataFrame:
    """One row per case, one column per shape feature."""
    return pd.DataFrame([shape_features(rows) for rows in cases], columns=SHAPE_FEATURES)


def train_shape_classifier(attempts: List[Attempt]) -> lightgbm.LGBMClassifier:
    """Fit the typology classifier on whole attempts (training period only)."""
    model = lightgbm.LGBMClassifier(
        n_estimators=TREES,
        num_leaves=LEAVES,
        learning_rate=LEARNING_RATE,
        min_child_samples=MIN_CASES_PER_LEAF,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(shape_table([a.rows for a in attempts]), np.array([a.typology for a in attempts]))
    return model


def predict_shapes(model: lightgbm.LGBMClassifier, cases: List[CaseRows]) -> List[str]:
    """The typology the classifier names for each case."""
    if not cases:
        return []
    return [str(name) for name in model.predict(shape_table(cases))]


def accuracy_of(truths: List[str], guesses: List[str]) -> float:
    """Share of cases named correctly."""
    if not truths:
        return 0.0
    return sum(1 for t, g in zip(truths, guesses) if t == g) / len(truths)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> List[float]:
    """95% range for an accuracy measured on `total` cases. Wide when there are few cases."""
    if total == 0:
        return [0.0, 0.0]
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [float(centre - half), float(centre + half)]


def per_class_scores(truths: List[str], guesses: List[str]) -> Dict[str, Dict[str, float]]:
    """Precision, recall, F1 and number of cases for each typology."""
    counts = Counter(truths)
    result = {}
    for label in TYPOLOGIES:
        score = score_label(truths, guesses, label)
        result[label] = {"precision": score.precision, "recall": score.recall, "f1": score.f1, "cases": counts[label]}
    return result


def confusion_matrix(truths: List[str], guesses: List[str]) -> Dict[str, Dict[str, int]]:
    """matrix[true typology][guessed typology] = number of cases."""
    matrix = {t: {g: 0 for g in TYPOLOGIES} for t in TYPOLOGIES}
    for truth, guess in zip(truths, guesses):
        if truth in matrix and guess in matrix[truth]:
            matrix[truth][guess] += 1
    return matrix


def evaluate_cases(truths: List[str], guesses: List[str]) -> Dict[str, object]:
    """Accuracy with its 95% range, macro-F1, per-class scores and the confusion matrix."""
    per_class = per_class_scores(truths, guesses)
    correct = sum(1 for t, g in zip(truths, guesses) if t == g)
    present = [label for label in TYPOLOGIES if per_class[label]["cases"] > 0]
    return {
        "cases": len(truths),
        "accuracy": accuracy_of(truths, guesses),
        "accuracy_range": wilson_interval(correct, len(truths)),
        "macro_f1": float(np.mean([per_class[label]["f1"] for label in present])) if present else 0.0,
        "per_class": per_class,
        "confusion": confusion_matrix(truths, guesses),
    }
