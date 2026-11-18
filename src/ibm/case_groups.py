"""Form cases: laundering attempts from the patterns file, or groups of flagged transactions.

Two ways to get a case:
  * An attempt is one "BEGIN LAUNDERING ATTEMPT" block of HI-Small_Patterns.txt. Its typology is known.
    This is the best case: every payment of the pattern is present. It gives the ceiling.
  * A flagged case is built from the model's flagged transactions only: two flagged payments are in
    the same case when they share an account and are at most 7 days apart. This is the real setting:
    some payments of a pattern are missing and unrelated flagged payments may join in.

Attempts are split like src/ibm/build_sft.py: by the time of their first payment (train is the earliest
60% of the timeline, validation the next 20%, test the last 20%), and an account used by an earlier
split's attempts can never appear in a later split's attempts.

Example:
    attempts = split_attempts(parse_patterns(), boundaries)
    cases = link_flagged_payments(frame.iloc[flagged_positions])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.ibm.case_shape import SECONDS_PER_HOUR, CaseRows
from src.ibm.train import temporal_split

LINK_WINDOW_DAYS = 7  # the same window the evidence and graph checks use
LINK_WINDOW_SECONDS = LINK_WINDOW_DAYS * 24 * SECONDS_PER_HOUR
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Attempt:
    """One laundering attempt: its typology, payments and the accounts it touches."""

    attempt_id: int
    typology: str
    rows: CaseRows
    start: pd.Timestamp
    accounts: frozenset


def account_names(banks: pd.Series, accounts: pd.Series) -> List[str]:
    """'bank/account' for each row."""
    return [str(b) + "/" + str(a) for b, a in zip(banks, accounts)]


def rows_from_payments(part: pd.DataFrame) -> CaseRows:
    """The payments of a case, from any table with the IBM columns."""
    senders = account_names(part["from_bank"], part["from_account"])
    receivers = account_names(part["to_bank"], part["to_account"])
    seconds = [t.timestamp() for t in part["timestamp"]]
    crossing = (part["from_bank"].astype(str) != part["to_bank"].astype(str)).tolist()
    return CaseRows(senders, receivers, seconds, part["amount_paid"].astype(float).tolist(), crossing)


def make_attempts(patterns: pd.DataFrame) -> List[Attempt]:
    """One Attempt per attempt id of the parsed patterns file."""
    attempts = []
    for attempt_id, part in patterns.groupby("attempt_id"):
        rows = rows_from_payments(part)
        accounts = frozenset(rows.senders) | frozenset(rows.receivers)
        attempts.append(
            Attempt(int(attempt_id), str(part["typology"].iloc[0]), rows, part["timestamp"].min(), accounts)
        )
    return attempts


def split_boundaries(frame: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """The times where train ends and validation ends, from the same cut as the transaction model."""
    train, validation, _ = temporal_split(len(frame))
    return frame["timestamp"].iloc[train.stop], frame["timestamp"].iloc[validation.stop]


def split_attempts(
    attempts: List[Attempt], boundaries: Tuple[pd.Timestamp, pd.Timestamp]
) -> Tuple[Dict[str, List[Attempt]], Dict[str, List[Attempt]]]:
    """Attempts by split. Returns (account-disjoint split, time-only split).

    Time-only means the split by first payment alone. The account-disjoint split also drops any attempt
    that uses an account of an earlier split's attempts, the same rule as build_sft.py."""
    by_start: Dict[str, List[Attempt]] = {name: [] for name in SPLIT_NAMES}
    for attempt in attempts:
        name = SPLIT_NAMES[int(attempt.start >= boundaries[0]) + int(attempt.start >= boundaries[1])]
        by_start[name].append(attempt)
    claimed: Set[str] = set()
    disjoint: Dict[str, List[Attempt]] = {}
    for name in SPLIT_NAMES:
        disjoint[name] = [a for a in by_start[name] if not (a.accounts & claimed)]
        for attempt in by_start[name]:
            claimed |= attempt.accounts
    return disjoint, by_start


def find_root(parent: List[int], item: int) -> int:
    """Union-find lookup with path shortening."""
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def link_flagged_payments(part: pd.DataFrame) -> List[List[int]]:
    """Group payments into cases. Returns lists of row numbers (0-based within `part`).

    For each account, its payments in time order are linked to the next one when the gap is at most 7 days."""
    parent = list(range(len(part)))
    seconds = np.array([t.timestamp() for t in part["timestamp"]])
    senders = account_names(part["from_bank"], part["from_account"])
    receivers = account_names(part["to_bank"], part["to_account"])
    last_seen: Dict[str, int] = {}
    for row in np.argsort(seconds, kind="stable"):
        for account in {senders[row], receivers[row]}:
            earlier = last_seen.get(account)
            if earlier is not None and seconds[row] - seconds[earlier] <= LINK_WINDOW_SECONDS:
                parent[find_root(parent, int(row))] = find_root(parent, earlier)
            last_seen[account] = int(row)
    groups: Dict[int, List[int]] = {}
    for row in range(len(part)):
        groups.setdefault(find_root(parent, row), []).append(row)
    return list(groups.values())
