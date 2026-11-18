"""Build the supervised fine-tuning examples from the IBM laundering rows.

Each example is an alert plus its counted evidence (the investigate.py format) and
a target: the typology from the patterns file and a short case note.

Splitting follows two rules at once:
  1. By time: train is the earliest 60% of the timeline, validation the next 20%, test the last 20%
     (the same cut as src/ibm/train.py).
  2. By account: an account that appears in a laundering row of an earlier period can never be
     used in a later split, so no account is in both train and test.
Typologies are balanced by taking up to a fixed number per typology, spread across attempts.
The NONE class is matched non-laundering payments from the same day and payment format.

Example:
    python -m src.ibm.build_sft          # writes data/ibm_sft/{train,val,test}.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.ibm.case_text import AlertFacts
from src.ibm.evidence import (
    Evidence,
    TransactionIndex,
    build_evidence,
    subset_for_alerts,
)
from src.ibm.investigate import account_name, facts_for, listed_laundering_positions
from src.ibm.loader import NOT_LAUNDERING, add_account_codes, load_labelled_transactions
from src.ibm.sft_format import (
    NO_PATTERN,
    TYPOLOGIES,
    build_input_text,
    find_answer_problems,
    target_answer,
)
from src.ibm.train import temporal_split

RANDOM_SEED = 42  # same seed as the rest of the project, so the files rebuild identically
OUTPUT_FOLDER = Path("data/ibm_sft")
PER_CLASS = {
    "train": 220,
    "val": 40,
    "test": 80,
}  # about 3,000 examples over nine classes
SPLIT_NAMES = (
    "train",
    "val",
    "test",
)  # in time order; earlier splits claim accounts first
NO_SCORE = 0.0  # AlertFacts needs a score; build_input_text never shows it


def accounts_of(frame: pd.DataFrame, positions: np.ndarray) -> Set[int]:
    """Every account code that sends or receives in these rows."""
    part = frame.iloc[positions]
    return set(part["from_code"].tolist()) | set(part["to_code"].tolist())


def without_claimed_accounts(frame: pd.DataFrame, positions: np.ndarray, claimed: Set[int]) -> np.ndarray:
    """Keep only rows whose sender and receiver no earlier split has used."""
    part = frame.iloc[positions]
    blocked = part["from_code"].isin(claimed) | part["to_code"].isin(claimed)
    return positions[~blocked.to_numpy()]


def spread_across_attempts(attempt_ids: List[int], limit: int, rng: random.Random) -> List[int]:
    """Indexes into `attempt_ids`, taking one row per attempt in turn so no one attempt dominates."""
    by_attempt: Dict[int, List[int]] = {}
    for place, attempt in enumerate(attempt_ids):
        by_attempt.setdefault(attempt, []).append(place)
    for places in by_attempt.values():
        rng.shuffle(places)
    chosen: List[int] = []
    queues = [by_attempt[key] for key in sorted(by_attempt)]
    while len(chosen) < limit and any(queues):
        for queue in queues:
            if queue and len(chosen) < limit:
                chosen.append(queue.pop())
    return chosen


def pick_laundering_rows(frame: pd.DataFrame, positions: np.ndarray, per_class: int, rng: random.Random) -> List[int]:
    """Up to `per_class` rows for each typology, spread across that typology's attempts."""
    picked: List[int] = []
    for typology in TYPOLOGIES:
        rows = [p for p in positions if frame["typology"].iat[p] == typology]
        attempts = [int(frame["attempt_id"].iat[p]) for p in rows]
        picked += [int(rows[i]) for i in spread_across_attempts(attempts, per_class, rng)]
    return picked


def negative_lookup(frame: pd.DataFrame, period: slice, claimed: Set[int]) -> Dict[tuple, np.ndarray]:
    """Unclaimed non-laundering rows of a period, grouped by (day, payment format)."""
    part = frame.iloc[period]
    quiet = (part["typology"] == NOT_LAUNDERING) & ~part["from_code"].isin(claimed) & ~part["to_code"].isin(claimed)
    rows = part[quiet]
    keys = list(zip(rows["timestamp"].dt.date, rows["payment_format"].astype(str)))
    lookup: Dict[tuple, List[int]] = {}
    start = period.start or 0
    for key, place in zip(keys, np.flatnonzero(quiet.to_numpy())):
        lookup.setdefault(key, []).append(int(place) + start)
    return {key: np.array(places) for key, places in lookup.items()}


def match_negatives(
    frame: pd.DataFrame,
    templates: List[int],
    lookup: Dict[tuple, np.ndarray],
    count: int,
    rng: random.Random,
) -> List[int]:
    """For `count` laundering rows, one non-laundering row from the same day and payment format."""
    used: Set[int] = set()
    matches: List[int] = []
    for template in rng.sample(templates, min(count, len(templates))):
        key = (
            frame["timestamp"].iat[template].date(),
            str(frame["payment_format"].iat[template]),
        )
        options = [int(p) for p in lookup.get(key, []) if int(p) not in used]
        if options:
            choice = rng.choice(options)
            used.add(choice)
            matches.append(choice)
    return matches


def make_example(frame: pd.DataFrame, index: TransactionIndex, position: int, split: str) -> dict:
    """One training example: the input text, the target answer and bookkeeping for the split checks."""
    txn_id = int(frame.index[position])
    row = frame.iloc[position]
    typology = str(row["typology"])
    label = NO_PATTERN if typology == NOT_LAUNDERING else typology
    evidence: Evidence = build_evidence(index, txn_id)
    facts: AlertFacts = facts_for(row, txn_id, NO_SCORE, label)
    text, answer = build_input_text(facts, evidence), target_answer(label, evidence)
    assert not find_answer_problems(json.loads(answer), text), "target note must be faithful to its input"
    return {
        "alert_id": facts.alert_id,
        "split": split,
        "typology": label,
        "attempt_id": int(row["attempt_id"]),
        "sender": account_name(row["from_bank"], row["from_account"]),
        "receiver": account_name(row["to_bank"], row["to_account"]),
        "time": facts.time,
        "input": text,
        "answer": answer,
    }


def choose_positions(
    frame: pd.DataFrame,
    split: str,
    period: slice,
    claimed: Set[int],
    rng: random.Random,
) -> List[int]:
    """Row positions for one split: balanced laundering rows plus matched non-laundering rows."""
    listed = without_claimed_accounts(frame, listed_laundering_positions(frame, period), claimed)
    laundering = pick_laundering_rows(frame, listed, PER_CLASS[split], rng)
    lookup = negative_lookup(frame, period, claimed)
    return laundering + match_negatives(frame, laundering, lookup, PER_CLASS[split], rng)


def accounts_to_claim(frame: pd.DataFrame, period: slice) -> Set[int]:
    """Accounts in any laundering row of a period (listed or not): later splits must avoid all of them."""
    part = frame.iloc[period]
    positions = np.flatnonzero((part["is_laundering"] == 1).to_numpy()) + (period.start or 0)
    return accounts_of(frame, positions)


def build_examples(frame: pd.DataFrame, seed: int = RANDOM_SEED) -> Dict[str, List[dict]]:
    """All examples, by split. `frame` needs account codes plus typology and attempt_id columns."""
    rng = random.Random(seed)
    periods = dict(zip(SPLIT_NAMES, temporal_split(len(frame))))
    claimed: Set[int] = set()
    chosen: Dict[str, List[int]] = {}
    for split in SPLIT_NAMES:
        chosen[split] = choose_positions(frame, split, periods[split], claimed, rng)
        # Also claim the chosen matched negatives: they are not laundering rows but are still in this split.
        claimed |= accounts_to_claim(frame, periods[split]) | accounts_of(frame, np.array(chosen[split]))
    everything = np.array(sorted(p for positions in chosen.values() for p in positions))
    index = TransactionIndex(subset_for_alerts(frame, everything))
    return {s: [make_example(frame, index, p, s) for p in chosen[s]] for s in SPLIT_NAMES}


def save_examples(examples: Dict[str, List[dict]], folder: Path = OUTPUT_FOLDER) -> None:
    """Write one JSON line per example into <folder>/<split>.jsonl."""
    folder.mkdir(parents=True, exist_ok=True)
    for split, rows in examples.items():
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (folder / ("%s.jsonl" % split)).write_text(text, encoding="utf-8")


def load_examples(split: str, folder: Path = OUTPUT_FOLDER) -> List[dict]:
    """Read one saved split."""
    path = folder / ("%s.jsonl" % split)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def describe(examples: Dict[str, List[dict]]) -> str:
    """Counts per split and typology, for the console."""
    lines: List[str] = []
    for split, rows in examples.items():
        counts: Dict[str, int] = {}
        for row in rows:
            counts[row["typology"]] = counts.get(row["typology"], 0) + 1
        lines.append("%-5s %5d  %s" % (split, len(rows), json.dumps(counts, sort_keys=True)))
    return "\n".join(lines)


def main(output_folder: Optional[Path] = None) -> None:
    """Load the IBM data, build the three splits and save them."""
    frame = add_account_codes(load_labelled_transactions())
    examples = build_examples(frame)
    save_examples(examples, output_folder or OUTPUT_FOLDER)
    print(describe(examples))


if __name__ == "__main__":
    main()
