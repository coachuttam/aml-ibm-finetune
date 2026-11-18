from __future__ import annotations

import pandas as pd

from src.ibm.hybrid import best_rule_for_typology, f1_for_typology, hybrid_flags
from tests.ibm_sample import make_transactions


def make_labelled_frame() -> pd.DataFrame:
    frame = make_transactions([{"timestamp": "2022/09/01 1%d:00" % hour} for hour in range(4)])
    frame["typology"] = ["UNLISTED", "UNLISTED", "NONE", "NONE"]
    return frame


def test_hybrid_flags_a_row_when_either_the_model_or_the_rule_flags_it():
    model = pd.Series([True, False, False])
    rule = pd.Series([False, False, True])
    assert list(hybrid_flags(model, rule)) == [True, False, True]


def test_f1_for_typology_is_one_when_exactly_that_typology_is_flagged():
    frame = make_labelled_frame()
    flagged = pd.Series([True, True, False, False])
    assert f1_for_typology(flagged, frame, slice(0, 4), "UNLISTED") == 1.0


def test_best_rule_for_typology_prefers_the_rule_that_finds_it_precisely():
    frame = make_labelled_frame()
    flags = pd.DataFrame(
        {
            "R1": [False, False, False, False],
            "R2": [True, True, True, True],
            "R3": [True, True, False, False],
            "R4": [False] * 4,
            "R5": [False] * 4,
        }
    )
    assert best_rule_for_typology(frame, flags, slice(0, 4), "UNLISTED") == "R3"
