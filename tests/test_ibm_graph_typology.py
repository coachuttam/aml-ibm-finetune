from __future__ import annotations

from src.ibm.evidence import TransactionIndex
from src.ibm.graph_typology import GraphConfig, Neighbourhood, classify_by_graph, hops_back
from tests.ibm_sample import make_transactions

CONFIG = GraphConfig(degree_min=3, max_cycle_hops=4)


def pay(hour, sender, receiver, amount=100.0):
    """One payment on 1 Sept; banks are fixed, accounts are the letters."""
    return {
        "timestamp": "2022/09/01 %02d:00" % hour,
        "from_account": sender,
        "to_account": receiver,
        "from_bank": "001",
        "to_bank": "001",
        "amount_paid": amount,
    }


def verdict_for(rows, last_txn_id):
    frame = make_transactions(rows)
    return classify_by_graph(TransactionIndex(frame), last_txn_id, CONFIG)


def test_fan_out_is_found_when_one_sender_pays_many_accounts():
    verdict = verdict_for([pay(h, "S", "R%d" % h) for h in range(4)], 3)
    assert verdict.typology == "FAN_OUT" and "4 different accounts" in verdict.reason


def test_fan_in_is_found_when_many_accounts_pay_one_receiver():
    verdict = verdict_for([pay(h, "P%d" % h, "R") for h in range(4)], 3)
    assert verdict.typology == "FAN_IN"


def test_cycle_is_found_when_money_comes_back_to_the_sender():
    verdict = verdict_for([pay(1, "A", "B"), pay(2, "B", "C"), pay(3, "C", "A"), pay(4, "A", "B")], 3)
    assert verdict.typology == "CYCLE" and "reaches the sender again" in verdict.reason


def test_scatter_gather_is_found_when_split_money_meets_at_one_account():
    rows = [
        pay(1, "S", "M1"),
        pay(2, "S", "M2"),
        pay(3, "S", "M3"),
        pay(4, "M1", "G"),
        pay(5, "M2", "G"),
        pay(6, "M3", "G"),
    ]
    assert verdict_for(rows, 5).typology == "SCATTER_GATHER"


def test_gather_scatter_is_found_when_an_account_collects_then_pays_out():
    rows = [
        pay(1, "P1", "H"),
        pay(2, "P2", "H"),
        pay(3, "P3", "H"),
        pay(4, "H", "Q1"),
        pay(5, "H", "Q2"),
        pay(6, "H", "Q3"),
    ]
    assert verdict_for(rows, 5).typology == "GATHER_SCATTER"


def test_bipartite_is_found_when_two_payers_pay_the_same_two_accounts():
    rows = [pay(1, "A", "X"), pay(2, "A", "Y"), pay(3, "B", "X"), pay(4, "B", "Y")]
    assert verdict_for(rows, 3).typology == "BIPARTITE"


def test_no_verdict_when_there_is_no_shape():
    assert verdict_for([pay(1, "A", "B")], 0) is None


def test_graph_checks_ignore_payments_after_the_alert():
    rows = [pay(1, "S", "R1"), pay(2, "S", "R2"), pay(3, "S", "R3"), pay(4, "S", "R4")]
    assert verdict_for(rows, 0) is None


def test_hops_back_counts_the_payments_on_the_shortest_path():
    frame = make_transactions([pay(1, "A", "B"), pay(2, "B", "C"), pay(3, "C", "D")])
    index = TransactionIndex(frame)
    hood = Neighbourhood(index, int(index.seconds[-1]))
    a, d = int(frame["from_code"].iloc[0]), int(frame["to_code"].iloc[2])
    assert hops_back(hood, a, d, 5) == 3
    assert hops_back(hood, a, d, 2) is None
