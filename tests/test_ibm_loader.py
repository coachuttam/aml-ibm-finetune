from __future__ import annotations

from src.ibm.loader import NOT_LAUNDERING, UNLISTED_TYPOLOGY, attach_typology, parse_patterns
from tests.ibm_sample import make_transactions, write_patterns_file


def test_patterns_file_gives_each_listed_transaction_its_typology(tmp_path):
    path = tmp_path / "patterns.txt"
    write_patterns_file(path)
    patterns = parse_patterns(path)
    assert list(patterns["typology"]) == ["FAN_OUT", "FAN_OUT"]
    assert list(patterns["attempt_id"]) == [0, 0]


def test_attach_typology_labels_listed_rows_and_leaves_others_alone(tmp_path):
    path = tmp_path / "patterns.txt"
    write_patterns_file(path)
    transactions = make_transactions(
        [
            {"timestamp": "2022/09/01 10:00", "is_laundering": 1},
            {
                "timestamp": "2022/09/01 11:00",
                "to_bank": "003",
                "to_account": "C",
                "amount_received": 200.0,
                "amount_paid": 200.0,
                "is_laundering": 1,
            },
            {"timestamp": "2022/09/01 12:00"},
        ]
    )
    labelled = attach_typology(transactions, parse_patterns(path))
    assert list(labelled["typology"]) == ["FAN_OUT", "FAN_OUT", NOT_LAUNDERING]


def test_laundering_rows_missing_from_the_patterns_file_are_marked_unlisted(tmp_path):
    path = tmp_path / "patterns.txt"
    write_patterns_file(path)
    transactions = make_transactions([{"timestamp": "2022/09/02 09:00", "is_laundering": 1}])
    labelled = attach_typology(transactions, parse_patterns(path))
    assert labelled["typology"].iloc[0] == UNLISTED_TYPOLOGY


def test_identical_repeated_transactions_are_matched_one_to_one(tmp_path):
    path = tmp_path / "patterns.txt"
    write_patterns_file(path)
    same_row = {"timestamp": "2022/09/01 10:00", "is_laundering": 1}
    labelled = attach_typology(make_transactions([same_row, same_row]), parse_patterns(path))
    assert sorted(labelled["typology"]) == ["FAN_OUT", UNLISTED_TYPOLOGY]


def test_account_codes_are_shared_between_sender_and_receiver_sides():
    frame = make_transactions(
        [
            {"from_bank": "001", "from_account": "A", "to_bank": "002", "to_account": "B"},
            {
                "timestamp": "2022/09/01 11:00",
                "from_bank": "002",
                "from_account": "B",
                "to_bank": "003",
                "to_account": "C",
            },
        ]
    )
    assert frame["to_code"].iloc[0] == frame["from_code"].iloc[1]
    assert frame["from_code"].iloc[0] != frame["to_code"].iloc[1]
