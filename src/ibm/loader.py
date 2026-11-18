"""Load the IBM AML transactions and attach a laundering typology to each one.

The transactions file only says "laundering: yes or no". The patterns file lists
every laundering transaction again, grouped under a header such as
"BEGIN LAUNDERING ATTEMPT - FAN-OUT", so joining the two tells us the typology.

Example:
    transactions = load_labelled_transactions()
    transactions["typology"].value_counts()
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

DATA_DIR = Path("data")
TRANSACTIONS_PATH = DATA_DIR / "HI-Small_Trans.csv"
PATTERNS_PATH = DATA_DIR / "HI-Small_Patterns.txt"

NOT_LAUNDERING = "NONE"
# Laundering rows that the patterns file does not describe still count as laundering.
UNLISTED_TYPOLOGY = "UNLISTED"

# The CSV header repeats "Account" twice, so we name the columns ourselves.
COLUMN_NAMES = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]
CATEGORY_COLUMNS = [
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "receiving_currency",
    "payment_currency",
    "payment_format",
]
# A transaction is identified by all of these together (the file has no id column).
MATCH_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "amount_paid",
    "payment_currency",
    "payment_format",
]
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
BEGIN_PATTERN = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)")


def load_transactions(path: Path = TRANSACTIONS_PATH, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read the CSV, sorted by time, with a `txn_id` that is the row's position in time order."""
    frame = pd.read_csv(
        path,
        header=0,
        names=COLUMN_NAMES,
        nrows=max_rows,
        dtype={
            "from_bank": str,
            "from_account": str,
            "to_bank": str,
            "to_account": str,
        },
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format=TIMESTAMP_FORMAT)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    for column in CATEGORY_COLUMNS:
        frame[column] = frame[column].astype("category")
    frame.index.name = "txn_id"
    return frame


def parse_patterns(path: Path = PATTERNS_PATH) -> pd.DataFrame:
    """Read the patterns file into one row per listed transaction, with its typology and attempt number."""
    rows: List[List[object]] = []
    typology = ""
    attempt_number = -1
    for line in path.read_text().splitlines():
        header = BEGIN_PATTERN.search(line)
        if header:
            typology = header.group(1).replace("-", "_")
            attempt_number += 1
        elif line.strip() and not line.startswith("END"):
            rows.append(line.split(",") + [typology, attempt_number])
    frame = pd.DataFrame(rows, columns=COLUMN_NAMES + ["typology", "attempt_id"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format=TIMESTAMP_FORMAT)
    for column in ("amount_received", "amount_paid"):
        frame[column] = frame[column].astype(float)
    return frame.drop(columns=["is_laundering"])


def _with_occurrence_number(frame: pd.DataFrame) -> pd.DataFrame:
    """Number identical transactions 0, 1, 2... so repeats can be matched one to one."""
    frame = frame.copy()
    for column in ("from_bank", "from_account", "to_bank", "to_account", "payment_currency", "payment_format"):
        frame[column] = frame[column].astype(str)
    frame["occurrence"] = frame.groupby(MATCH_COLUMNS).cumcount()
    return frame


def attach_typology(transactions: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Add `typology` and `attempt_id` columns. Rows outside any pattern get NONE (or UNLISTED if flagged)."""
    candidates = transactions[transactions["timestamp"].isin(patterns["timestamp"])]
    left = _with_occurrence_number(candidates.reset_index())
    right = _with_occurrence_number(patterns)
    keys = MATCH_COLUMNS + ["occurrence"]
    matched = left.merge(right[keys + ["typology", "attempt_id"]], on=keys, how="left")
    result = transactions.copy()
    result["typology"] = NOT_LAUNDERING
    result["attempt_id"] = -1
    result.loc[matched["txn_id"].values, "typology"] = matched["typology"].fillna(NOT_LAUNDERING).values
    result.loc[matched["txn_id"].values, "attempt_id"] = matched["attempt_id"].fillna(-1).astype(int).values
    unlisted = (result["is_laundering"] == 1) & (result["typology"] == NOT_LAUNDERING)
    result.loc[unlisted, "typology"] = UNLISTED_TYPOLOGY
    return result


def load_labelled_transactions(max_rows: Optional[int] = None) -> pd.DataFrame:
    """Transactions with typology labels, ready for rules and features."""
    return attach_typology(load_transactions(max_rows=max_rows), parse_patterns())


def add_account_codes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `from_code` and `to_code`: one whole number per (bank, account), the same on both sides.

    Rules and features group by these numbers because integers are much faster than text.
    """
    sender_names = frame["from_bank"].astype(str) + "/" + frame["from_account"].astype(str)
    receiver_names = frame["to_bank"].astype(str) + "/" + frame["to_account"].astype(str)
    codes, _ = pd.factorize(pd.concat([sender_names, receiver_names], ignore_index=True))
    result = frame.copy()
    result["from_code"] = codes[: len(frame)]
    result["to_code"] = codes[len(frame) :]
    return result
