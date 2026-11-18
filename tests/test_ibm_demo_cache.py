from __future__ import annotations

from ibm_view import alert_label, merge_with_demo
from src.ibm.demo_cache import load_demo_records, pick_demo_records, reason_line, write_cache


def record(alert_id, typology, truth, source="graph", score=1.0):
    return {
        "alert_id": alert_id,
        "typology": typology,
        "true_typology": truth,
        "typology_source": source,
        "score": score,
        "reason": "The sender paid 9 different accounts in 7 days.",
    }


RECORDS = [
    record("IBM-1", "FAN_IN", "FAN_IN"),
    record("IBM-2", "FAN_OUT", "FAN_OUT"),
    record("IBM-3", "GATHER_SCATTER", "NONE", "classifier"),
    record("IBM-4", "CYCLE", "CYCLE"),
]


def test_picks_choose_the_best_ranked_match_for_each_role():
    picks = pick_demo_records(RECORDS)
    assert {role: item["alert_id"] for role, item in picks.items()} == {
        "clear fan-out": "IBM-2",
        "cycle": "IBM-4",
        "borderline": "IBM-3",
    }


def test_a_role_with_no_matching_alert_is_left_out_not_invented():
    assert "cycle" not in pick_demo_records(RECORDS[:2])


def test_a_classifier_guess_does_not_count_as_a_clear_match():
    guesses = [record("IBM-9", "FAN_OUT", "FAN_OUT", source="classifier")]
    assert "clear fan-out" not in pick_demo_records(guesses)


def test_cache_round_trips_in_role_order_and_writes_the_index(tmp_path):
    picks = pick_demo_records(RECORDS)
    ranks = {"clear fan-out": 2, "cycle": 4, "borderline": 3}
    write_cache(picks, ranks, tmp_path / "cache", tmp_path / "demo.md")
    loaded = load_demo_records(tmp_path / "cache")
    assert [item["demo_role"] for item in loaded] == ["clear fan-out", "cycle", "borderline"]
    text = (tmp_path / "demo.md").read_text()
    assert "`IBM-2`" in text and "rank 2" in text


def test_load_demo_records_is_empty_when_nothing_is_cached(tmp_path):
    assert load_demo_records(tmp_path / "missing") == []


def test_borderline_reason_says_it_is_a_mismatch():
    assert "mismatch" in reason_line("borderline", RECORDS[2], 3)


def test_the_app_lists_demo_alerts_first_without_duplicates():
    demo = [dict(RECORDS[1], demo_role="clear fan-out")]
    merged = merge_with_demo(demo, [RECORDS[0], RECORDS[1]])
    assert [item["alert_id"] for item in merged] == ["IBM-2", "IBM-1"]
    assert alert_label(merged[0]) == "IBM-2 (demo: clear fan-out)"
