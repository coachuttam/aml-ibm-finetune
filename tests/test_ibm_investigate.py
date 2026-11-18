from __future__ import annotations

import numpy as np
import pytest

from src.case_note import GuardrailError
from src.case_store import AWAITING_ANALYST, load_case
from src.ibm.case_text import AlertFacts, build_findings, draft_case_note, draft_report
from src.ibm.evidence import TransactionIndex, build_evidence, subset_for_alerts
from src.ibm.investigate import (
    investigate_alert,
    listed_laundering_positions,
    load_records,
    run_investigations,
    save_records,
    top_positions,
    typology_accuracy,
)
from tests.ibm_sample import make_transactions


def fan_out_frame():
    """Account A pays five different accounts in five hours; one unrelated payment comes after."""
    rows = [
        {"timestamp": "2022/09/01 0%d:00" % hour, "to_account": "R%d" % hour, "amount_paid": 500.0 + hour}
        for hour in range(5)
    ]
    rows.append({"timestamp": "2022/09/01 09:00", "from_account": "Z", "to_account": "Y"})
    frame = make_transactions(rows)
    frame["typology"] = ["FAN_OUT"] * 5 + ["NONE"]
    return frame


def evidence_for(frame, txn_id):
    index = TransactionIndex(subset_for_alerts(frame, np.array([txn_id])))
    return build_evidence(index, txn_id)


def facts_for_test(txn_id=4, typology="FAN_OUT"):
    return AlertFacts(
        "IBM-%d" % txn_id, txn_id, "2022/09/01 04:00", "001/A", "002/R4", 504.0, "US Dollar", 0.97, typology
    )


def test_evidence_counts_only_what_happened_up_to_the_alert():
    frame = fan_out_frame()
    numbers = evidence_for(frame, 2).numbers
    assert numbers["sender_sent_count_7d"] == 3
    assert numbers["sender_distinct_receivers_7d"] == 3


def test_evidence_ignores_unrelated_accounts():
    frame = fan_out_frame()
    assert evidence_for(frame, 5).numbers["sender_sent_count_7d"] == 1


def test_velocity_counts_the_last_hour_and_the_last_day():
    numbers = evidence_for(fan_out_frame(), 4).numbers
    assert numbers["sender_sent_count_1h"] == 2
    assert numbers["sender_sent_count_24h"] == 5


def test_findings_use_the_counted_numbers_and_plain_plurals():
    evidence = evidence_for(fan_out_frame(), 4)
    findings = build_findings(facts_for_test(), evidence)
    assert any("5 payments to 5 different accounts" in line for line in findings)
    assert not any("1 payments" in line for line in findings)


def test_case_note_and_report_name_the_typology_and_say_nothing_was_filed():
    evidence = evidence_for(fan_out_frame(), 4)
    findings = build_findings(facts_for_test(), evidence)
    note = draft_case_note(facts_for_test(), findings, evidence)
    report = draft_report(facts_for_test(), findings)
    assert "Fan-out" in note and "Fan-out" in report
    assert "Nothing has been filed or sent" in note and "not filed" in report


def test_the_wording_check_refuses_accusing_words_in_the_report():
    facts = facts_for_test()
    with pytest.raises(GuardrailError):
        draft_report(facts, ["This looks like fraud."])


def test_investigate_alert_saves_a_case_that_waits_for_the_analyst(tmp_path):
    frame = fan_out_frame()
    record = investigate_alert(facts_for_test(), evidence_for(frame, 4), tmp_path / "cases", tmp_path / "log.jsonl")
    case = load_case("IBM-4", tmp_path / "cases")
    assert case.status == AWAITING_ANALYST and case.decision is None
    assert [step.number for step in case.steps] == [1, 2, 3, 4, 5, 6]
    assert case.steps[5].status == "waiting"
    assert record["status"] == AWAITING_ANALYST
    assert (tmp_path / "log.jsonl").exists()


def test_an_unknown_typology_sends_the_case_to_review(tmp_path):
    record = investigate_alert(
        facts_for_test(typology="UNKNOWN"), evidence_for(fan_out_frame(), 4), tmp_path / "cases", tmp_path / "log.jsonl"
    )
    assert record["status"] == "needs_review"


def test_top_positions_returns_the_best_scores_inside_the_period():
    scores = np.array([0.9, 0.1, 0.8, 0.95, 0.2])
    assert list(top_positions(scores, slice(1, 5), count=2)) == [3, 2]


def test_listed_laundering_positions_skip_normal_and_unlisted_rows():
    frame = fan_out_frame()
    frame["typology"] = ["FAN_OUT", "NONE", "UNLISTED", "CYCLE", "NONE", "NONE"]
    assert list(listed_laundering_positions(frame, slice(0, 6))) == [0, 3]


def test_typology_accuracy_counts_only_rows_with_a_listed_typology():
    result = typology_accuracy(["FAN_OUT", "NONE", "CYCLE", "UNLISTED"], ["FAN_OUT", "FAN_OUT", "STACK", "STACK"])
    assert result == {"rows": 2, "correct": 1, "accuracy": 0.5}


def test_records_round_trip_through_the_jsonl_file(tmp_path):
    path = tmp_path / "records.jsonl"
    assert load_records(path) is None
    save_records([{"alert_id": "IBM-1", "typology": "FAN_OUT"}], path)
    assert load_records(path) == [{"alert_id": "IBM-1", "typology": "FAN_OUT"}]


def test_run_investigations_works_end_to_end_on_a_tiny_frame(tmp_path):
    frame = fan_out_frame()
    frame = frame.iloc[:6]
    scores = np.linspace(0.1, 0.9, len(frame))
    result = run_investigations(frame, scores, count=2, case_folder=tmp_path / "c", log_path=tmp_path / "log.jsonl")
    assert len(result["records"]) == 2
    assert {"top_accuracy", "test_accuracy", "top_really_laundering"} <= set(result)


def test_out_of_fold_guesses_have_one_guess_per_alert():
    from src.ibm.typology import out_of_fold_typologies

    frame = fan_out_frame()
    evidence = [evidence_for(frame, i) for i in range(5)]
    guesses = out_of_fold_typologies(evidence, ["FAN_OUT", "CYCLE", "FAN_OUT", "CYCLE", "FAN_OUT"], folds=2)
    assert len(guesses) == 5
