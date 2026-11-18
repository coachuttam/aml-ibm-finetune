from __future__ import annotations

from src.ibm.rules import (
    FAN_MIN_COUNTERPARTIES,
    flag_fan_in,
    flag_fan_out,
    flag_foreign_exchange_hops,
    flag_pass_through,
    flag_structuring,
    run_rules,
    score_flags,
)
from tests.ibm_sample import make_transactions


def test_structuring_flags_three_payments_just_under_the_limit_in_one_day():
    frame = make_transactions([{"amount_paid": 9500.0}] * 3 + [{"amount_paid": 9500.0, "from_account": "Z"}])
    flags = flag_structuring(frame)
    assert flags.sum() == 3


def test_structuring_ignores_two_payments_and_amounts_over_the_limit():
    frame = make_transactions([{"amount_paid": 9500.0}] * 2 + [{"amount_paid": 12000.0}] * 3)
    assert flag_structuring(frame).sum() == 0


def test_fan_out_flags_an_account_paying_many_different_accounts_in_one_day():
    rows = [{"to_account": "R%d" % number} for number in range(FAN_MIN_COUNTERPARTIES)]
    assert flag_fan_out(make_transactions(rows)).all()


def test_fan_out_does_not_flag_repeat_payments_to_the_same_account():
    rows = [{"to_account": "R"}] * (FAN_MIN_COUNTERPARTIES + 2)
    assert not flag_fan_out(make_transactions(rows)).any()


def test_fan_in_flags_many_different_accounts_paying_one_account():
    rows = [{"from_account": "S%d" % number} for number in range(FAN_MIN_COUNTERPARTIES)]
    assert flag_fan_in(make_transactions(rows)).all()


def test_pass_through_flags_the_send_that_forwards_what_arrived():
    frame = make_transactions(
        [
            {"from_account": "X", "to_account": "M", "amount_paid": 5000.0},
            {
                "timestamp": "2022/09/01 12:00",
                "from_bank": "002",
                "from_account": "M",
                "to_account": "Y",
                "amount_paid": 4900.0,
            },
        ]
    )
    flags = flag_pass_through(frame)
    assert list(flags) == [False, True]


def test_pass_through_ignores_tiny_amounts():
    frame = make_transactions(
        [
            {"from_account": "X", "to_account": "M", "amount_paid": 0.01},
            {
                "timestamp": "2022/09/01 12:00",
                "from_bank": "002",
                "from_account": "M",
                "to_account": "Y",
                "amount_paid": 0.01,
            },
        ]
    )
    assert not flag_pass_through(frame).any()


def test_fx_hop_flags_money_sent_on_in_a_currency_it_did_not_arrive_in():
    frame = make_transactions(
        [
            {"from_account": "X", "to_account": "M", "receiving_currency": "Euro", "payment_currency": "Euro"},
            {
                "timestamp": "2022/09/01 12:00",
                "from_bank": "002",
                "from_account": "M",
                "to_account": "Y",
                "payment_currency": "Yen",
                "receiving_currency": "Yen",
            },
        ]
    )
    assert list(flag_foreign_exchange_hops(frame)) == [False, True]


def test_run_rules_has_one_column_per_rule_and_an_any_rule_column():
    flags = run_rules(make_transactions([{}]))
    assert list(flags.columns) == ["R1", "R2", "R3", "R4", "R5", "any_rule"]


def test_score_flags_reports_precision_and_recall():
    frame = make_transactions([{"is_laundering": 1}, {"is_laundering": 1}, {"is_laundering": 0}, {"is_laundering": 0}])
    flags = frame["is_laundering"].astype(bool)
    flags.iloc[1] = False
    flags.iloc[2] = True
    score = score_flags(flags, frame["is_laundering"])
    assert score == {"alerts": 2, "precision": 0.5, "recall": 0.5}
