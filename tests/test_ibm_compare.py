from __future__ import annotations

import pandas as pd

from src.ibm.compare import markdown_table, recall_by_typology, rule_scores_for
from src.ibm.rules import run_rules
from tests.ibm_sample import make_transactions


def test_markdown_table_has_a_header_a_divider_and_one_line_per_row():
    text = markdown_table(["A", "B"], [["1", "2"], ["3", "4"]])
    assert text.splitlines() == ["| A | B |", "|---|---|", "| 1 | 2 |", "| 3 | 4 |"]


def test_recall_by_typology_is_the_flagged_share_of_each_typology():
    frame = make_transactions([{}, {"timestamp": "2022/09/01 11:00"}, {"timestamp": "2022/09/01 12:00"}])
    frame["typology"] = ["FAN_OUT", "FAN_OUT", "NONE"]
    flagged = pd.Series([True, False, True], index=frame.index)
    assert recall_by_typology(frame, flagged, slice(0, 3)) == {"FAN_OUT": 0.5}


def test_rule_scores_have_a_row_for_every_rule_and_any_rule():
    frame = make_transactions([{}])
    rows = rule_scores_for(frame, run_rules(frame), slice(0, 1))
    assert [row[0] for row in rows] == ["R1", "R2", "R3", "R4", "R5", "any_rule"]
