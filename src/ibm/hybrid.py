"""Hybrid: flag a transaction if the model scores it high OR the best rule for UNLISTED fires.

The model misses the laundering rows that the patterns file does not describe
(typology UNLISTED), while the rules catch some of them. We pick the rule that
does best on UNLISTED in the training period only, then OR it with the model.

Example:
    rule_id = best_rule_for_typology(frame, flags, train_period, "UNLISTED")
    hybrid = hybrid_flags(model_flags, flags[rule_id])
"""

from __future__ import annotations

import pandas as pd

from src.ibm.rules import RULES


def f1_for_typology(flagged: pd.Series, frame: pd.DataFrame, period: slice, typology: str) -> float:
    """F1 of one yes/no column at finding one typology's transactions (other rows count as misses)."""
    flagged_in_period = flagged.iloc[period]
    wanted = frame["typology"].iloc[period] == typology
    hits = int((flagged_in_period & wanted).sum())
    precision = hits / max(int(flagged_in_period.sum()), 1)
    recall = hits / max(int(wanted.sum()), 1)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def best_rule_for_typology(frame: pd.DataFrame, flags: pd.DataFrame, period: slice, typology: str) -> str:
    """The rule id with the best F1 on `typology` in `period`. F1 (not recall) stops a noisy rule winning."""
    best_id, best_score = RULES[0].rule_id, -1.0
    for rule in RULES:
        score = f1_for_typology(flags[rule.rule_id], frame, period, typology)
        if score > best_score:
            best_id, best_score = rule.rule_id, score
    return best_id


def hybrid_flags(model_flags: pd.Series, rule_flags: pd.Series) -> pd.Series:
    """True where the model flagged the row or the rule did."""
    return model_flags | rule_flags
