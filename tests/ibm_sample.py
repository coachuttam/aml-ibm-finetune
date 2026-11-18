"""Tiny hand-made IBM-style transactions for the tests (not real data)."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from src.ibm.loader import COLUMN_NAMES, add_account_codes


def make_transactions(rows: List[dict]) -> pd.DataFrame:
    """Build a frame from partial rows; missing fields get plain defaults."""
    defaults = {
        "timestamp": "2022/09/01 10:00",
        "from_bank": "001",
        "from_account": "A",
        "to_bank": "002",
        "to_account": "B",
        "amount_received": 100.0,
        "receiving_currency": "US Dollar",
        "amount_paid": 100.0,
        "payment_currency": "US Dollar",
        "payment_format": "ACH",
        "is_laundering": 0,
    }
    full_rows = [dict(defaults, **row) for row in rows]
    frame = pd.DataFrame(full_rows, columns=COLUMN_NAMES)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="%Y/%m/%d %H:%M")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    frame.index.name = "txn_id"
    return add_account_codes(frame)


def write_patterns_file(path, blocks: Optional[List[str]] = None) -> None:
    """Write a small patterns file: one fan-out attempt with two transactions."""
    text = (
        blocks[0]
        if blocks
        else (
            "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out\n"
            "2022/09/01 10:00,001,A,002,B,100.00,US Dollar,100.00,US Dollar,ACH,1\n"
            "2022/09/01 11:00,001,A,003,C,200.00,US Dollar,200.00,US Dollar,ACH,1\n"
            "END LAUNDERING ATTEMPT - FAN-OUT\n"
        )
    )
    path.write_text(text)
