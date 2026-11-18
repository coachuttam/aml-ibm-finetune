"""Investigate the model's top IBM alerts: evidence, case note, draft report, typology score.

The steps follow the main project's fixed order (see src/investigator.py) and use
its Case and StepRecord containers, audit log and wording check. The main
investigator itself is built around rupee and income features, so it is not
called here. Steps 1 to 5 only write local files; step 6 waits for the analyst.

Example:
    records = run_investigations(frame, scores, case_folder=Path("audit/cases"))
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.audit import AUDIT_LOG, append_event
from src.case_note import GuardrailError
from src.case_store import AWAITING_ANALYST, CASE_FOLDER, NEEDS_REVIEW_STATUS, Case, StepRecord, save_case
from src.ibm.case_text import AlertFacts, build_findings, draft_case_note, draft_report, typology_words
from src.ibm.evidence import Evidence, TransactionIndex, build_evidence
from src.ibm.graph_typology import choose_config, classify_by_graph
from src.ibm.loader import NOT_LAUNDERING, UNLISTED_TYPOLOGY
from src.ibm.train import temporal_split
from src.ibm.typology import out_of_fold_typologies, predict_typologies, train_typology_classifier

TOP_ALERT_COUNT = 20
RECORDS_PATH = Path("results/ibm_investigations.jsonl")
UNKNOWN_TYPOLOGY = "UNKNOWN"
WITHHELD = "Withheld: the wording check failed. Review by hand."


def account_name(bank: object, account: object) -> str:
    return "%s/%s" % (bank, account)


def facts_for(frame_row: pd.Series, txn_id: int, score: float, typology: str, reason: str = "") -> AlertFacts:
    """The alert's own fields, ready for the note."""
    return AlertFacts(
        alert_id="IBM-%d" % txn_id,
        txn_id=txn_id,
        time=str(frame_row["timestamp"]),
        sender=account_name(frame_row["from_bank"], frame_row["from_account"]),
        receiver=account_name(frame_row["to_bank"], frame_row["to_account"]),
        amount=float(frame_row["amount_paid"]),
        currency=str(frame_row["payment_currency"]),
        score=float(score),
        typology=typology,
        reason=reason,
    )


def build_steps(facts: AlertFacts, evidence: Evidence, findings: List[str], note_ok: bool) -> List[StepRecord]:
    """The six workflow steps as the main project records them."""
    rows = evidence.sender_rows + evidence.receiver_rows
    times = sorted(row["time"] for row in rows)
    known = facts.typology != UNKNOWN_TYPOLOGY
    return [
        StepRecord(1, "Load alert", "done", "Loaded %s." % facts.alert_id, ["Model score %.3f" % facts.score]),
        StepRecord(
            2,
            "Fetch transactions",
            "done" if rows else "unavailable",
            "Fetched %d evidence rows." % len(rows),
            ["From %s to %s." % (times[0], times[-1])] if rows else [],
        ),
        StepRecord(3, "Compute findings", "done", "Computed %d findings." % len(findings), findings),
        StepRecord(
            4,
            "Classify",
            "done" if known else "needs_review",
            "Suggested typology: %s." % typology_words(facts.typology),
            [],
        ),
        StepRecord(
            5,
            "Draft case note and report",
            "done" if note_ok else "needs_review",
            "Drafted the case note and a draft report." if note_ok else WITHHELD,
            [],
        ),
        StepRecord(6, "Analyst decision", "waiting", "Waiting for the analyst. Nothing has been filed or sent.", []),
    ]


def investigate_alert(
    facts: AlertFacts, evidence: Evidence, case_folder: Path = CASE_FOLDER, log_path: Path = AUDIT_LOG
) -> Dict[str, object]:
    """Run steps 1 to 5 for one alert, save the case file, log each step, return a plain record."""
    findings = build_findings(facts, evidence)
    try:
        note = draft_case_note(facts, findings, evidence)
        report = draft_report(facts, findings)
    except GuardrailError:
        note, report = WITHHELD, WITHHELD
    steps = build_steps(facts, evidence, findings, note_ok=note != WITHHELD)
    review = any(step.status == "needs_review" for step in steps)
    case = Case(
        facts.alert_id, NEEDS_REVIEW_STATUS if review else AWAITING_ANALYST, steps, note + "\n\n---\n\n" + report
    )
    save_case(case, case_folder)
    for step in steps[:5]:
        append_event(
            facts.alert_id,
            "step %d: %s" % (step.number, step.name.lower()),
            after="%s: %s" % (step.status, step.summary),
            log_path=log_path,
        )
    return {
        **asdict(facts),
        "status": case.status,
        "numbers": evidence.numbers,
        "sender_rows": evidence.sender_rows,
        "receiver_rows": evidence.receiver_rows,
        "findings": findings,
        "case_note": note,
        "report": report,
        "steps": [asdict(s) for s in steps],
    }


def top_positions(scores: np.ndarray, period: slice, count: int = TOP_ALERT_COUNT) -> np.ndarray:
    """Row positions of the highest-scored transactions in a period, best first."""
    start = period.start or 0
    order = np.argsort(-scores[period], kind="stable")[:count]
    return order + start


def listed_laundering_positions(frame: pd.DataFrame, period: slice) -> np.ndarray:
    """Positions in a period of laundering rows whose typology the patterns file names."""
    part = frame.iloc[period]
    listed = ~part["typology"].isin([NOT_LAUNDERING, UNLISTED_TYPOLOGY])
    return np.flatnonzero(listed.to_numpy()) + (period.start or 0)


def collect_evidence(frame: pd.DataFrame, positions: np.ndarray, index: TransactionIndex) -> Dict[int, Evidence]:
    """Evidence for each position, keyed by transaction id."""
    return {int(frame.index[p]): build_evidence(index, int(frame.index[p])) for p in positions}


def typology_accuracy(truths: List[str], guesses: List[str]) -> Dict[str, float]:
    """Share of guesses that match the patterns file, over rows that have a listed typology."""
    pairs = [(t, g) for t, g in zip(truths, guesses) if t not in (NOT_LAUNDERING, UNLISTED_TYPOLOGY)]
    correct = sum(1 for t, g in pairs if t == g)
    return {"rows": len(pairs), "correct": correct, "accuracy": correct / len(pairs) if pairs else 0.0}


def combine_guesses(index, txn_ids, config, fallback_guesses):
    """Graph verdict first; the classifier's guess when no graph check fires. Returns (typology, reason, source)."""
    combined = []
    for txn_id, backup in zip(txn_ids, fallback_guesses):
        verdict = classify_by_graph(index, txn_id, config)
        if verdict is None:
            combined.append((backup, "", "classifier"))
        else:
            combined.append((verdict.typology, verdict.reason, "graph"))
    return combined


def run_investigations(
    frame: pd.DataFrame,
    scores: np.ndarray,
    count: int = TOP_ALERT_COUNT,
    case_folder: Path = CASE_FOLDER,
    log_path: Path = AUDIT_LOG,
) -> Dict[str, object]:
    """Investigate the top test alerts with graph checks (classifier as fallback) and score the typologies."""
    train, _, test = temporal_split(len(frame))
    train_positions = listed_laundering_positions(frame, train)
    test_positions = listed_laundering_positions(frame, test)
    top = top_positions(scores, test, count)
    index = TransactionIndex(frame)
    evidence = collect_evidence(frame, np.concatenate([train_positions, test_positions, top]), index)
    train_ids = [int(frame.index[p]) for p in train_positions]
    train_truths = [frame.at[i, "typology"] for i in train_ids]
    classifier = train_typology_classifier([evidence[i] for i in train_ids], train_truths)
    # Tune on out-of-fold guesses, so the fallback is as honest on training rows as it will be on test rows.
    fallback = out_of_fold_typologies([evidence[i] for i in train_ids], train_truths)
    config = choose_config(index, train_ids, train_truths, fallback)
    top_ids = [int(frame.index[p]) for p in top]
    test_ids = [int(frame.index[p]) for p in test_positions]
    scored = {}
    for name, ids in (("top", top_ids), ("test", test_ids)):
        backups = predict_typologies(classifier, [evidence[i] for i in ids], UNKNOWN_TYPOLOGY)
        scored[name] = (backups, combine_guesses(index, ids, config, backups))
    records = build_records(frame, scores, top, evidence, scored["top"][1], case_folder, log_path)
    return summarise(frame, records, test_ids, scored, config)


def summarise(frame, records, test_ids, scored, config) -> Dict[str, object]:
    """Old (classifier only) and new (graph plus fallback) typology accuracy, for the top alerts and all of test."""
    top_truths = [r["true_typology"] for r in records]
    test_truths = [frame.at[i, "typology"] for i in test_ids]
    result: Dict[str, object] = {"records": records, "graph_config": asdict(config)}
    for name, truths in (("top", top_truths), ("test", test_truths)):
        backups, combined = scored[name]
        result["classifier_%s_accuracy" % name] = typology_accuracy(truths, backups)
        result["%s_accuracy" % name] = typology_accuracy(truths, [guess[0] for guess in combined])
        result["%s_graph_share" % name] = sum(1 for guess in combined if guess[2] == "graph") / max(len(combined), 1)
    result["top_really_laundering"] = sum(1 for r in records if r["true_typology"] != NOT_LAUNDERING)
    return result


def build_records(frame, scores, top, evidence, guesses, case_folder, log_path) -> List[Dict[str, object]]:
    """Investigate each top alert and attach the true typology for scoring."""
    ids = [int(frame.index[p]) for p in top]
    records = []
    for position, txn_id, (typology, reason, source) in zip(top, ids, guesses):
        facts = facts_for(frame.iloc[position], txn_id, scores[position], typology, reason)
        record = investigate_alert(facts, evidence[txn_id], case_folder, log_path)
        record["true_typology"] = str(frame.at[txn_id, "typology"])
        record["typology_source"] = source
        records.append(record)
    return records


def save_records(records: List[Dict[str, object]], path: Path = RECORDS_PATH) -> None:
    """Write one JSON line per investigation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def load_records(path: Path = RECORDS_PATH) -> Optional[List[Dict[str, object]]]:
    """Read the saved investigations, or None if the file does not exist yet."""
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
