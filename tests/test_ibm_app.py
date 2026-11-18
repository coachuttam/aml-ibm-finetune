"""Open the IBM alerts tab in the real Streamlit app (no browser) and check what it shows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

from src.ibm.investigate import run_investigations, save_records
from tests.test_ibm_investigate import fan_out_frame

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")
TAB_NAME = "IBM alerts"
SELECTOR_KEY = "ibm_alert_choice"


def make_records(folder: Path) -> list:
    frame = fan_out_frame()
    scores = np.linspace(0.1, 0.9, len(frame))
    result = run_investigations(frame, scores, count=3, case_folder=folder / "cases", log_path=folder / "log.jsonl")
    save_records(result["records"], folder / "results" / "ibm_investigations.jsonl")
    return result["records"]


def open_app(monkeypatch, folder: Path) -> AppTest:
    monkeypatch.chdir(folder)
    return AppTest.from_file(APP_FILE, default_timeout=120).run()


def test_the_ibm_tab_exists_and_the_app_runs_without_errors(tmp_path, monkeypatch):
    make_records(tmp_path)
    app = open_app(monkeypatch, tmp_path)
    assert not app.exception
    assert TAB_NAME in [tab.label for tab in app.tabs]


def test_selecting_the_first_alert_shows_evidence_case_note_and_report(tmp_path, monkeypatch):
    records = make_records(tmp_path)
    app = open_app(monkeypatch, tmp_path)
    app.selectbox(key=SELECTOR_KEY).select(records[0]["alert_id"]).run()
    assert not app.exception
    text = "\n".join(block.value for block in app.markdown)
    assert records[0]["findings"][0] in text  # evidence: the first computed finding
    assert "# Case note: %s" % records[0]["alert_id"] in text  # case note
    assert "DRAFT suspicious transaction report (not filed)" in text  # draft report
    assert len(app.dataframe) >= 2  # sender and receiver evidence tables


def test_the_tab_says_what_to_run_when_there_are_no_investigations_yet(tmp_path, monkeypatch):
    app = open_app(monkeypatch, tmp_path)
    assert not app.exception
    assert any("python -m src.ibm.compare" in info.value for info in app.info)
