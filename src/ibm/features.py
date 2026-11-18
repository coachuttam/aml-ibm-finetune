"""Per-transaction features for the model, all computed from the past only.

For each transaction we describe the sender's (and receiver's) behaviour up to
that moment, never after it, so the model cannot peek at the future.
Amounts are in each row's own currency; the data has no exchange rates.

Example:
    table = build_features(transactions)   # needs account codes (see loader)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

SECONDS_PER_HOUR = 3600
VELOCITY_WINDOWS: Dict[str, int] = {
    "1h": SECONDS_PER_HOUR,
    "24h": 24 * SECONDS_PER_HOUR,
    "7d": 7 * 24 * SECONDS_PER_HOUR,
}
# Larger than any timestamp plus window, so one account's block never overlaps the next.
ACCOUNT_BLOCK_SIZE = 10**10
ROUND_AMOUNT_STEP = 100  # multiples of 100 look "made up" more often than real prices
VERY_ROUND_AMOUNT_STEP = 1_000

FEATURE_COLUMNS: List[str] = (
    ["sender_sent_%s" % name for name in VELOCITY_WINDOWS]
    + ["receiver_received_%s" % name for name in VELOCITY_WINDOWS]
    + [
        "sender_out_degree",
        "receiver_in_degree",
        "sender_in_out_ratio",
        "is_round_amount",
        "is_very_round_amount",
        "is_cross_bank",
        "is_currency_change",
        "log_amount_paid",
        "payment_format_code",
        "payment_currency_code",
    ]
)


def _seconds(frame: pd.DataFrame) -> np.ndarray:
    return (frame["timestamp"] - frame["timestamp"].min()).dt.total_seconds().to_numpy().astype(np.int64)


def cumulative_up_to(
    stream_codes: np.ndarray,
    stream_seconds: np.ndarray,
    stream_values: np.ndarray,
    query_codes: np.ndarray,
    query_seconds: np.ndarray,
) -> np.ndarray:
    """For each query, the total of `stream_values` for the same account at or before that second.

    Example: a stream of two payments by account 7 (values 1 and 1, at seconds 10 and 50)
    answers a query (account 7, second 60) with 2.0.
    """
    stream_keys = stream_codes.astype(np.int64) * ACCOUNT_BLOCK_SIZE + stream_seconds
    order = np.argsort(stream_keys, kind="stable")
    sorted_keys = stream_keys[order]
    running_total = np.concatenate([[0.0], np.cumsum(stream_values[order].astype(float))])
    query_keys = query_codes.astype(np.int64) * ACCOUNT_BLOCK_SIZE + query_seconds
    position = np.searchsorted(sorted_keys, query_keys, side="right")
    block_start = np.searchsorted(sorted_keys, query_codes.astype(np.int64) * ACCOUNT_BLOCK_SIZE, side="left")
    return running_total[position] - running_total[block_start]


def add_velocity(frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """How many payments the sender made, and the receiver got, in the last 1 hour, 24 hours, 7 days."""
    seconds = _seconds(frame)
    ones = np.ones(len(frame))
    from_codes, to_codes = frame["from_code"].to_numpy(), frame["to_code"].to_numpy()
    for name, window in VELOCITY_WINDOWS.items():
        now_sent = cumulative_up_to(from_codes, seconds, ones, from_codes, seconds)
        before_sent = cumulative_up_to(from_codes, seconds, ones, from_codes, seconds - window - 1)
        table["sender_sent_%s" % name] = now_sent - before_sent
        now_received = cumulative_up_to(to_codes, seconds, ones, to_codes, seconds)
        before_received = cumulative_up_to(to_codes, seconds, ones, to_codes, seconds - window - 1)
        table["receiver_received_%s" % name] = now_received - before_received


def _running_distinct_partners(own_codes: pd.Series, partner_codes: pd.Series) -> np.ndarray:
    """For each row, how many different partners that account has dealt with so far."""
    first_time_pair = ~pd.DataFrame({"own": own_codes, "partner": partner_codes}).duplicated()
    return first_time_pair.groupby(own_codes).cumsum().to_numpy()


def add_degree(frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """Different accounts the sender has paid so far, and different accounts that paid the receiver so far."""
    table["sender_out_degree"] = _running_distinct_partners(frame["from_code"], frame["to_code"])
    table["receiver_in_degree"] = _running_distinct_partners(frame["to_code"], frame["from_code"])


def add_in_out_ratio(frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """Money the sender has sent so far divided by money it has received so far (high = pass-through)."""
    seconds = _seconds(frame)
    amounts = frame["amount_paid"].to_numpy()
    from_codes = frame["from_code"].to_numpy()
    sent = cumulative_up_to(from_codes, seconds, amounts, from_codes, seconds)
    received = cumulative_up_to(frame["to_code"].to_numpy(), seconds, amounts, from_codes, seconds)
    table["sender_in_out_ratio"] = sent / (received + 1.0)


def add_simple_flags(frame: pd.DataFrame, table: pd.DataFrame) -> None:
    """Round amounts, cross-bank, currency change, and the plain amount and format columns."""
    amount = frame["amount_paid"]
    table["is_round_amount"] = ((amount % ROUND_AMOUNT_STEP == 0) & (amount > 0)).astype(int)
    table["is_very_round_amount"] = ((amount % VERY_ROUND_AMOUNT_STEP == 0) & (amount > 0)).astype(int)
    table["is_cross_bank"] = (frame["from_bank"].astype(str) != frame["to_bank"].astype(str)).astype(int)
    table["is_currency_change"] = (
        frame["receiving_currency"].astype(str) != frame["payment_currency"].astype(str)
    ).astype(int)
    table["log_amount_paid"] = np.log1p(amount)
    table["payment_format_code"] = frame["payment_format"].astype("category").cat.codes
    table["payment_currency_code"] = frame["payment_currency"].astype("category").cat.codes


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The full feature table, one row per transaction, in the same order as `frame`."""
    table = pd.DataFrame(index=frame.index)
    add_velocity(frame, table)
    add_degree(frame, table)
    add_in_out_ratio(frame, table)
    add_simple_flags(frame, table)
    return table[FEATURE_COLUMNS]
