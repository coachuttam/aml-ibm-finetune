"""Tests for case shape counting, case grouping and case scoring."""

from __future__ import annotations

import pandas as pd

from src.ibm.case_flagged import FlaggedCase, accuracy_by_case_size
from src.ibm.case_groups import Attempt, link_flagged_payments, split_attempts
from src.ibm.case_shape import CaseRows, shape_features
from src.ibm.case_typology import wilson_interval


def rows_from(pairs, amounts=None):
    senders = [p[0] for p in pairs]
    receivers = [p[1] for p in pairs]
    count = len(pairs)
    return CaseRows(senders, receivers, [3600.0 * i for i in range(count)], amounts or [100.0] * count, [False] * count)


def test_a_fan_out_has_one_sender_and_many_receivers_and_no_cycle():
    features = shape_features(rows_from([("A", "B"), ("A", "C"), ("A", "D")]))
    assert features["max_out_degree"] == 3
    assert features["sources"] == 1 and features["sinks"] == 3
    assert features["has_cycle"] == 0 and features["layers"] == 1


def test_a_cycle_is_found_with_its_shortest_length():
    features = shape_features(rows_from([("A", "B"), ("B", "C"), ("C", "A")]))
    assert features["has_cycle"] == 1
    assert features["shortest_cycle"] == 3


def test_a_stack_is_deep_and_has_middlemen():
    features = shape_features(rows_from([("A", "B"), ("B", "C"), ("C", "D")]))
    assert features["layers"] == 3
    assert features["middle_nodes"] == 2


def test_two_senders_paying_two_receivers_is_bipartite_and_fully_joined():
    features = shape_features(rows_from([("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")]))
    assert features["bipartite_flag"] == 1
    assert features["bipartite_density"] == 1.0


def test_a_payment_to_oneself_is_not_an_edge_and_equal_amounts_have_no_spread():
    features = shape_features(rows_from([("A", "A"), ("A", "B")]))
    assert features["edges"] == 1
    assert features["amount_cv"] == 0.0


def payments(rows):
    frame = pd.DataFrame(rows, columns=["from_account", "to_account", "timestamp"])
    frame["from_bank"] = "1"
    frame["to_bank"] = "1"  # one bank, so an account has one name whether it sends or receives
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def test_flagged_payments_that_share_an_account_within_seven_days_join_one_case():
    part = payments([("A", "B", "2022-09-01"), ("B", "C", "2022-09-05"), ("X", "Y", "2022-09-03")])
    sizes = sorted(len(group) for group in link_flagged_payments(part))
    assert sizes == [1, 2]


def test_flagged_payments_more_than_seven_days_apart_stay_separate():
    part = payments([("A", "B", "2022-09-01"), ("B", "C", "2022-09-20")])
    assert len(link_flagged_payments(part)) == 2


def attempt(attempt_id, start, accounts):
    return Attempt(attempt_id, "FAN_OUT", rows_from([("A", "B")]), pd.Timestamp(start), frozenset(accounts))


def test_split_never_lets_an_attempt_reuse_an_account_from_an_earlier_split():
    attempts = [
        attempt(1, "2022-09-01", ["A", "B"]),
        attempt(2, "2022-09-10", ["B", "C"]),
        attempt(3, "2022-09-10", ["D"]),
    ]
    boundaries = (pd.Timestamp("2022-09-05"), pd.Timestamp("2022-09-08"))
    disjoint, time_only = split_attempts(attempts, boundaries)
    assert [a.attempt_id for a in disjoint["test"]] == [3]
    assert len(time_only["test"]) == 2


def test_case_truth_is_the_most_common_listed_typology_and_false_alarms_have_none():
    assert FlaggedCase([1, 2, 3], ["CYCLE", "CYCLE", "NONE"], "CYCLE").truth == "CYCLE"
    assert FlaggedCase([1], ["NONE"], "CYCLE").truth is None


def test_accuracy_by_case_size_ignores_cases_without_a_listed_typology():
    cases = [FlaggedCase([1], ["CYCLE"], "CYCLE"), FlaggedCase([2], ["NONE"], "CYCLE")]
    result = accuracy_by_case_size(cases)
    assert result["1 payment"] == {"cases": 1, "accuracy": 1.0}


def test_the_accuracy_range_is_wider_with_fewer_cases():
    small = wilson_interval(9, 10)
    large = wilson_interval(90, 100)
    assert (small[1] - small[0]) > (large[1] - large[0])
