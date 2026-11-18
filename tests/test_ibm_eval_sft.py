"""Tests for scoring the IBM typology models."""

from __future__ import annotations

from src.ibm.eval_sft import accuracy, macro_f1_over, predicted_typology, score_system


def example(alert_id: str, typology: str) -> dict:
    return {"alert_id": alert_id, "typology": typology}


def test_an_unknown_or_missing_answer_counts_as_invalid():
    assert predicted_typology(None) == "INVALID"
    assert predicted_typology({"typology": "FRAUD"}) == "INVALID"
    assert predicted_typology({"typology": "CYCLE"}) == "CYCLE"


def test_accuracy_is_the_share_of_exact_matches():
    assert accuracy(["A", "B", "A", "B"], ["A", "B", "B", "B"]) == 0.75


def test_macro_f1_counts_a_label_that_is_never_predicted_as_zero():
    assert macro_f1_over(["A", "B"], ["A", "A"], ["A", "B"]) < 0.5


def test_score_system_separates_laundering_rows_from_non_laundering_rows():
    examples = [example("1", "FAN_OUT"), example("2", "NONE")]
    answers = {
        "1": {"typology": "FAN_OUT", "valid": True, "problems": []},
        "2": {"typology": "FAN_OUT", "valid": False, "problems": ["note uses a number that is not in the input"]},
    }
    scores = score_system(examples, answers)
    assert scores["accuracy_laundering"] == 1.0
    assert scores["accuracy_all"] == 0.5
    assert scores["faithful_notes"] == 0.5
    assert scores["valid_answers"] == 0.5
