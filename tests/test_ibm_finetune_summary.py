from __future__ import annotations

from src.ibm.finetune_summary import accuracy, build_summary, score_system


def test_accuracy_is_the_share_of_exact_typology_matches():
    examples = [{"label": {"typology": "STRUCTURING"}}, {"label": {"typology": "NO_CONCERN"}}]
    assert accuracy(examples, ["STRUCTURING", "STRUCTURING"]) == 0.5
    assert accuracy([], []) == 0.0


def test_score_system_returns_none_when_there_is_no_answers_file(tmp_path):
    assert score_system("test", "base", tmp_path) is None


def test_summary_marks_missing_answer_files_instead_of_failing(tmp_path):
    text = build_summary(tmp_path)
    assert "no answers" in text and "Fine-tuned" in text


def test_real_saved_answers_give_scores_between_zero_and_one():
    score = score_system("test", "finetuned")
    assert score is not None and 0.0 <= score["accuracy"] <= 1.0 and 0.0 <= score["macro_f1"] <= 1.0
