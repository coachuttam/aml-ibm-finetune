"""Evidence for one IBM alert: what the sender and receiver did in the 7 days up to the alert.

Python counts everything here. The numbers feed the case note, the draft report
and the typology classifier. Only rows at or before the alert's time are used.

Example:
    index = TransactionIndex(subset_for_alerts(frame, positions))
    evidence = build_evidence(index, txn_id)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600
HISTORY_DAYS = 7  # the window the task asks for; long enough to see a whole short pattern
HISTORY_SECONDS = HISTORY_DAYS * 24 * SECONDS_PER_HOUR
MAX_ROWS_SHOWN = 10  # evidence rows kept per side, so a hub account does not flood the case note

EPOCH = pd.Timestamp("2000-01-01")  # any fixed start works; only differences in time matter

NUMBER_NAMES: List[str] = [
    "sender_sent_count_7d",
    "sender_received_count_7d",
    "sender_distinct_receivers_7d",
    "sender_distinct_payers_7d",
    "sender_sent_total_7d",
    "sender_received_total_7d",
    "sender_sent_count_1h",
    "sender_sent_count_24h",
    "sender_currencies_7d",
    "sender_cross_bank_share_7d",
    "receiver_received_count_7d",
    "receiver_distinct_payers_7d",
    "receiver_sent_count_7d",
    "receiver_distinct_receivers_7d",
    "receiver_paid_sender_back_7d",
    "alert_amount",
]


@dataclass
class Evidence:
    """Counted facts about one alert and the transaction rows behind them."""

    txn_id: int
    numbers: Dict[str, float]
    sender_rows: List[dict]
    receiver_rows: List[dict]


def subset_for_alerts(frame: pd.DataFrame, alert_positions: np.ndarray) -> pd.DataFrame:
    """Only the rows that touch an account named in an alert (much smaller than the full table)."""
    alerts = frame.iloc[alert_positions]
    accounts = np.unique(np.concatenate([alerts["from_code"].to_numpy(), alerts["to_code"].to_numpy()]))
    touches = frame["from_code"].isin(accounts) | frame["to_code"].isin(accounts)
    return frame[touches]


class TransactionIndex:
    """Fast "what did this account send or receive between two times" lookups on a time-sorted table."""

    def __init__(self, subset: pd.DataFrame) -> None:
        self.subset = subset
        self.seconds = (subset["timestamp"] - EPOCH).dt.total_seconds().to_numpy().astype(np.int64)
        self.sent_by = subset.groupby("from_code").indices
        self.received_by = subset.groupby("to_code").indices

    def position_of(self, txn_id: int) -> int:
        """Where a transaction id sits in the table."""
        return int(self.subset.index.get_loc(txn_id))

    def window(self, lookup: Dict[int, np.ndarray], code: int, end: int, length: int) -> np.ndarray:
        """Positions of this account's rows with time in [end - length, end]."""
        positions = lookup.get(code, np.array([], dtype=int))
        times = self.seconds[positions]
        first = np.searchsorted(times, end - length, side="left")
        last = np.searchsorted(times, end, side="right")
        return positions[first:last]


def describe_row(subset: pd.DataFrame, position: int) -> dict:
    """One transaction as a plain dict, for display."""
    row = subset.iloc[position]
    return {
        "txn_id": int(subset.index[position]),
        "time": str(row["timestamp"]),
        "from": "%s/%s" % (row["from_bank"], row["from_account"]),
        "to": "%s/%s" % (row["to_bank"], row["to_account"]),
        "amount": float(row["amount_paid"]),
        "currency": str(row["payment_currency"]),
        "format": str(row["payment_format"]),
    }


def count_numbers(
    index: TransactionIndex, sender: int, receiver: int, end: int, alert_amount: float
) -> Dict[str, float]:
    """All the counted numbers for one alert (see NUMBER_NAMES)."""
    sub = index.subset
    sent = index.window(index.sent_by, sender, end, HISTORY_SECONDS)
    got = index.window(index.received_by, sender, end, HISTORY_SECONDS)
    receiver_got = index.window(index.received_by, receiver, end, HISTORY_SECONDS)
    receiver_sent = index.window(index.sent_by, receiver, end, HISTORY_SECONDS)
    sent_rows = sub.iloc[sent]
    paid_back = sub.iloc[receiver_sent]["to_code"].to_numpy() == sender
    return {
        "sender_sent_count_7d": len(sent),
        "sender_received_count_7d": len(got),
        "sender_distinct_receivers_7d": sent_rows["to_code"].nunique(),
        "sender_distinct_payers_7d": sub.iloc[got]["from_code"].nunique(),
        "sender_sent_total_7d": float(sent_rows["amount_paid"].sum()),
        "sender_received_total_7d": float(sub.iloc[got]["amount_paid"].sum()),
        "sender_sent_count_1h": len(index.window(index.sent_by, sender, end, SECONDS_PER_HOUR)),
        "sender_sent_count_24h": len(index.window(index.sent_by, sender, end, 24 * SECONDS_PER_HOUR)),
        "sender_currencies_7d": sent_rows["payment_currency"].astype(str).nunique(),
        "sender_cross_bank_share_7d": _cross_bank_share(sent_rows),
        "receiver_received_count_7d": len(receiver_got),
        "receiver_distinct_payers_7d": sub.iloc[receiver_got]["from_code"].nunique(),
        "receiver_sent_count_7d": len(receiver_sent),
        "receiver_distinct_receivers_7d": sub.iloc[receiver_sent]["to_code"].nunique(),
        "receiver_paid_sender_back_7d": int(paid_back.sum()),
        "alert_amount": float(alert_amount),
    }


def _cross_bank_share(rows: pd.DataFrame) -> float:
    if rows.empty:
        return 0.0
    return float((rows["from_bank"].astype(str) != rows["to_bank"].astype(str)).mean())


def build_evidence(index: TransactionIndex, txn_id: int) -> Evidence:
    """Evidence for the transaction with this id."""
    position = index.position_of(txn_id)
    row = index.subset.iloc[position]
    sender, receiver, end = int(row["from_code"]), int(row["to_code"]), int(index.seconds[position])
    numbers = count_numbers(index, sender, receiver, end, row["amount_paid"])
    sender_rows = index.window(index.sent_by, sender, end, HISTORY_SECONDS)[-MAX_ROWS_SHOWN:]
    receiver_rows = index.window(index.received_by, receiver, end, HISTORY_SECONDS)[-MAX_ROWS_SHOWN:]
    return Evidence(
        txn_id,
        numbers,
        [describe_row(index.subset, p) for p in sender_rows],
        [describe_row(index.subset, p) for p in receiver_rows],
    )
