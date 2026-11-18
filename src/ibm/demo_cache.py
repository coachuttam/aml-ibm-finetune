"""Pick three demo alerts and cache their full investigations so the app can show them instantly.

Roles: a clear fan-out and a clear cycle (the graph check and the patterns file agree), and a
borderline alert (a high-scoring alert that is not laundering, so the typology is a mismatch).
The search runs down the model's ranked test alerts, because the top 20 hold no fan-out or cycle.

Example:
    python -m src.ibm.demo_cache
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from src.ibm.investigate import run_investigations
from src.ibm.loader import NOT_LAUNDERING
from src.ibm.train import fit_model, prepare_data, temporal_split

CACHE_FOLDER = Path("results/demo_cache")
SUMMARY_PATH = Path("results/demo_alerts.md")
SEARCH_DEPTH = 2000  # the model raises 1,655 alerts at its threshold; look a little past that
BORDERLINE_DEPTH = 20  # the borderline alert comes from the top 20 that the main report investigates
ROLES = ("clear fan-out", "cycle", "borderline")


def first_match(records: List[dict], wanted_typology: str) -> Optional[dict]:
    """The best-ranked alert where the graph check and the patterns file both say `wanted_typology`."""
    for record in records:
        if record["typology"] == wanted_typology == record["true_typology"] and record["typology_source"] == "graph":
            return record
    return None


def first_false_alarm(records: List[dict]) -> Optional[dict]:
    """The best-ranked alert in the top 20 that is not laundering at all."""
    for record in records[:BORDERLINE_DEPTH]:
        if record["true_typology"] == NOT_LAUNDERING:
            return record
    return None


def pick_demo_records(records: List[dict]) -> Dict[str, dict]:
    """Choose the three demo alerts from records sorted best score first. Roles with no match are left out."""
    picks = {
        "clear fan-out": first_match(records, "FAN_OUT"),
        "cycle": first_match(records, "CYCLE"),
        "borderline": first_false_alarm(records),
    }
    return {role: record for role, record in picks.items() if record is not None}


def reason_line(role: str, record: dict, rank: int) -> str:
    """One plain sentence on why this alert was picked."""
    base = "rank %d of the model's test alerts, score %.3f; " % (rank, record["score"])
    if role == "borderline":
        mismatch = "the model flags it but the patterns file says it is not laundering, "
        return base + mismatch + "so the suggested typology (%s) is a mismatch." % record["typology"]
    return base + "the graph check (%s) and the patterns file agree." % record["reason"]


def write_cache(
    picks: Dict[str, dict], ranks: Dict[str, int], folder: Path = CACHE_FOLDER, summary_path: Path = SUMMARY_PATH
) -> None:
    """One JSON file per demo alert (evidence, case note, draft report) plus the markdown index."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Demo alerts",
        "",
        "Pre-run investigations are cached in `results/demo_cache/`; nothing is filed or sent.",
        "",
    ]
    for role, record in picks.items():
        cached = dict(record, demo_role=role)
        (folder / ("%s.json" % record["alert_id"])).write_text(json.dumps(cached, ensure_ascii=False, indent=2))
        lines.append("- **%s**: `%s`. %s" % (role, record["alert_id"], reason_line(role, record, ranks[role])))
    summary_path.write_text("\n".join(lines) + "\n")


def load_demo_records(folder: Path = CACHE_FOLDER) -> List[dict]:
    """The cached demo investigations in role order, or an empty list if none are cached."""
    if not folder.exists():
        return []
    records = [json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))]
    return sorted(records, key=lambda record: ROLES.index(record["demo_role"]))


def main() -> None:
    frame, features = prepare_data()
    truth = frame["is_laundering"].to_numpy()
    train, validation, _ = temporal_split(len(frame))
    scores = fit_model(features, truth, train, validation, 1.0).predict_proba(features)[:, 1]
    with tempfile.TemporaryDirectory() as scratch:
        result = run_investigations(frame, scores, SEARCH_DEPTH, Path(scratch) / "cases", Path(scratch) / "log.jsonl")
    records = result["records"]
    picks = pick_demo_records(records)
    ranks = {role: records.index(record) + 1 for role, record in picks.items()}
    write_cache(picks, ranks)
    print({role: record["alert_id"] for role, record in picks.items()})


if __name__ == "__main__":
    main()
