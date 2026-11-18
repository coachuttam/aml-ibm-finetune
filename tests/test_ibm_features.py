from __future__ import annotations

import numpy as np

from src.ibm.features import FEATURE_COLUMNS, build_features, cumulative_up_to
from tests.ibm_sample import make_transactions


def test_cumulative_up_to_only_counts_the_same_account_up_to_that_second():
    codes = np.array([7, 7, 8])
    seconds = np.array([10, 50, 20])
    values = np.array([1.0, 1.0, 1.0])
    answer = cumulative_up_to(codes, seconds, values, np.array([7, 7, 8]), np.array([30, 60, 5]))
    assert list(answer) == [1.0, 2.0, 0.0]


def test_build_features_returns_every_named_column_for_every_row():
    table = build_features(make_transactions([{}, {"timestamp": "2022/09/01 11:00"}]))
    assert list(table.columns) == FEATURE_COLUMNS
    assert len(table) == 2


def test_sender_velocity_counts_only_payments_inside_the_window():
    frame = make_transactions(
        [
            {"timestamp": "2022/09/01 00:00"},
            {"timestamp": "2022/09/01 00:30"},
            {"timestamp": "2022/09/01 05:00"},
        ]
    )
    table = build_features(frame)
    assert list(table["sender_sent_1h"]) == [1, 2, 1]
    assert list(table["sender_sent_24h"]) == [1, 2, 3]


def test_velocity_never_counts_later_payments():
    frame = make_transactions([{"timestamp": "2022/09/01 00:00"}, {"timestamp": "2022/09/03 00:00"}])
    assert build_features(frame)["sender_sent_7d"].iloc[0] == 1


def test_degree_counts_new_partners_only_once():
    frame = make_transactions(
        [
            {"to_account": "B"},
            {"timestamp": "2022/09/01 11:00", "to_account": "B"},
            {"timestamp": "2022/09/01 12:00", "to_account": "C"},
        ]
    )
    assert list(build_features(frame)["sender_out_degree"]) == [1, 1, 2]


def test_in_out_ratio_is_high_for_an_account_that_forwards_what_it_received():
    frame = make_transactions(
        [
            {"from_account": "X", "to_account": "M", "amount_paid": 1000.0},
            {
                "timestamp": "2022/09/01 12:00",
                "from_bank": "002",
                "from_account": "M",
                "to_account": "Y",
                "amount_paid": 900.0,
            },
        ]
    )
    table = build_features(frame)
    assert table["sender_in_out_ratio"].iloc[1] > 0.8
    assert table["sender_in_out_ratio"].iloc[0] == 1000.0 / 1.0


def test_round_amount_and_cross_bank_flags():
    frame = make_transactions(
        [
            {"amount_paid": 5000.0, "to_bank": "002"},
            {"timestamp": "2022/09/01 11:00", "amount_paid": 5012.34, "to_bank": "001"},
        ]
    )
    table = build_features(frame)
    assert list(table["is_round_amount"]) == [1, 0]
    assert list(table["is_very_round_amount"]) == [1, 0]
    assert list(table["is_cross_bank"]) == [1, 0]
