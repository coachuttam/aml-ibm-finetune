"""Tests for the IBM fine-tuning example builder and the shared prompt format."""

from __future__ import annotations

import json
import random

import numpy as np

from src.ibm.build_sft import (
    accounts_of,
    build_examples,
    spread_across_attempts,
    without_claimed_accounts,
)
from src.ibm.case_text import AlertFacts
from src.ibm.evidence import Evidence
from src.ibm.sft_format import (
    build_input_text,
    find_answer_problems,
    target_answer,
    typology_of_answer,
)
from tests.ibm_sample import make_transactions

EVIDENCE = Evidence(
    1,
    {
        "sender_sent_count_7d": 9,
        "sender_received_count_7d": 0,
        "sender_distinct_receivers_7d": 9,
        "sender_distinct_payers_7d": 0,
        "sender_sent_total_7d": 900.0,
        "sender_received_total_7d": 0.0,
        "sender_sent_count_1h": 1,
        "sender_sent_count_24h": 3,
        "sender_currencies_7d": 1,
        "sender_cross_bank_share_7d": 1.0,
        "receiver_received_count_7d": 1,
        "receiver_distinct_payers_7d": 1,
        "receiver_sent_count_7d": 0,
        "receiver_distinct_receivers_7d": 0,
        "receiver_paid_sender_back_7d": 0,
        "alert_amount": 100.0,
    },
    [],
    [],
)
FACTS = AlertFacts(
    "IBM-1",
    1,
    "2022-09-01 10:00:00",
    "001/A",
    "002/B",
    100.0,
    "US Dollar",
    0.0,
    "FAN_OUT",
)


def test_the_input_text_never_shows_a_model_score_or_the_graph_reason():
    text = build_input_text(FACTS, EVIDENCE)
    assert "model score" not in text
    assert "Pattern check" not in text


def test_the_target_note_only_uses_numbers_that_are_in_the_input():
    text = build_input_text(FACTS, EVIDENCE)
    answer = json.loads(target_answer("FAN_OUT", EVIDENCE))
    assert find_answer_problems(answer, text) == []


def test_an_answer_with_an_invented_number_is_flagged():
    text = build_input_text(FACTS, EVIDENCE)
    answer = {"typology": "FAN_OUT", "note": "The sender paid 4242 accounts."}
    assert "note uses a number that is not in the input" in find_answer_problems(answer, text)


def test_an_answer_that_accuses_is_flagged_and_unknown_labels_are_flagged():
    answer = {"typology": "FRAUD", "note": "This is fraud."}
    problems = find_answer_problems(answer, "text")
    assert "note uses accusing wording" in problems
    assert "typology is not one of the allowed labels" in problems


def test_text_that_is_not_json_gives_no_answer():
    assert typology_of_answer("no braces here", "text") is None


def test_spreading_across_attempts_takes_one_row_per_attempt_in_turn():
    chosen = spread_across_attempts([1, 1, 1, 1, 2, 3], 3, random.Random(0))
    assert sorted(chosen_attempts([1, 1, 1, 1, 2, 3], chosen)) == [1, 2, 3]


def chosen_attempts(attempts, chosen):
    return [attempts[place] for place in chosen]


def test_claimed_accounts_are_removed_from_later_splits():
    frame = make_transactions([{"from_account": "A"}, {"from_account": "C", "to_account": "D"}])
    claimed = accounts_of(frame, np.array([0]))
    kept = without_claimed_accounts(frame, np.array([0, 1]), claimed)
    assert list(kept) == [1]


def test_split_never_puts_an_account_in_two_sets():
    frame = make_transactions(laundering_rows())
    frame["typology"] = ["FAN_OUT" if flag else "NONE" for flag in frame["is_laundering"]]
    frame["attempt_id"] = [i if flag else -1 for i, flag in enumerate(frame["is_laundering"])]
    examples = build_examples(frame)
    seen = {}
    for split, rows in examples.items():
        for row in rows:
            for account in (row["sender"], row["receiver"]):
                assert seen.setdefault(account, split) == split


def laundering_rows():
    """Ten days of small fan-outs from different senders, plus ordinary payments."""
    rows = []
    for day in range(1, 11):
        stamp = "2022/09/%02d 10:00" % day
        rows.append(
            {
                "timestamp": stamp,
                "from_account": "S%d" % day,
                "to_account": "R%d" % day,
                "is_laundering": 1,
            }
        )
        rows.append({"timestamp": stamp, "from_account": "N%d" % day, "to_account": "M%d" % day})
    return rows


class FakeTokenizer:
    """Turns each character into a token id, so token counts are easy to predict."""

    eos_token = "!"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(letter) for letter in text]}


def test_training_loss_covers_only_the_answer_tokens():
    from src.finetune import IGNORE_INDEX
    from src.ibm.finetune_ibm import tokenize_example

    example = {"alert_id": "IBM-1", "input": "abc", "answer": "xy"}
    item = tokenize_example(example, FakeTokenizer(), max_length=10_000)
    ignored = [label for label in item["labels"] if label == IGNORE_INDEX]
    assert len(item["labels"]) - len(ignored) == len("xy!")


def test_an_example_that_is_too_long_is_refused_rather_than_cut():
    import pytest

    from src.ibm.finetune_ibm import tokenize_example

    example = {"alert_id": "IBM-1", "input": "abc", "answer": "xy"}
    with pytest.raises(ValueError):
        tokenize_example(example, FakeTokenizer(), max_length=5)


def test_batch_size_grows_with_gpu_memory_and_keeps_an_effective_batch_of_16():
    from src.ibm.finetune_ibm import EFFECTIVE_BATCH_SIZE, pick_batch_size

    assert (pick_batch_size(15.0), pick_batch_size(24.0), pick_batch_size(40.0)) == (2, 4, 8)
    for memory in (15.0, 24.0, 40.0, 80.0):
        assert EFFECTIVE_BATCH_SIZE % pick_batch_size(memory) == 0
