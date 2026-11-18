"""Leakage audit: the model may only see past behaviour, never ids, labels or future rows."""

from __future__ import annotations

import pandas as pd

from src.ibm.features import FEATURE_COLUMNS, build_features
from tests.ibm_sample import make_transactions

FORBIDDEN_WORDS = ["account", "bank_id", "txn_id", "typology", "attempt", "laundering", "label"]
# These two are small whole numbers for the payment format and currency, not account identifiers.
ALLOWED_CODE_COLUMNS = ["payment_format_code", "payment_currency_code"]


def sample_history() -> pd.DataFrame:
    rows = []
    for hour in range(8):
        rows.append(
            {
                "timestamp": "2022/09/01 %02d:00" % hour,
                "to_account": "R%d" % (hour % 3),
                "amount_paid": 100.0 * (hour + 1),
            }
        )
    rows.append(
        {
            "timestamp": "2022/09/01 09:00",
            "from_account": "R0",
            "from_bank": "002",
            "to_account": "Z",
            "amount_paid": 250.0,
        }
    )
    return make_transactions(rows)


def test_no_feature_is_named_after_an_account_id_or_a_label():
    for name in FEATURE_COLUMNS:
        for word in FORBIDDEN_WORDS:
            assert word not in name, "%s looks like an id or label" % name


def test_only_the_payment_format_and_currency_codes_are_code_columns():
    code_columns = [name for name in FEATURE_COLUMNS if name.endswith("_code")]
    assert sorted(code_columns) == sorted(ALLOWED_CODE_COLUMNS)


def test_feature_table_has_exactly_the_listed_columns():
    table = build_features(sample_history())
    assert list(table.columns) == FEATURE_COLUMNS


def test_features_do_not_change_when_the_labels_change():
    frame = sample_history()
    relabelled = frame.copy()
    relabelled["is_laundering"] = 1
    relabelled["typology"] = "FAN_OUT"
    pd.testing.assert_frame_equal(build_features(frame), build_features(relabelled))


def test_features_do_not_change_when_account_ids_are_renamed():
    frame = sample_history()
    renamed = frame.copy()
    renamed["from_code"] = renamed["from_code"] + 1000
    renamed["to_code"] = renamed["to_code"] + 1000
    pd.testing.assert_frame_equal(build_features(frame), build_features(renamed))


def test_features_of_earlier_rows_ignore_every_later_row():
    frame = sample_history()
    first_part = frame.iloc[:5]
    with_future = build_features(frame).iloc[:5]
    without_future = build_features(first_part)
    pd.testing.assert_frame_equal(with_future, without_future)
