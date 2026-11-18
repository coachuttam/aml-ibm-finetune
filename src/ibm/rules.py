"""Five hand-written rules for the IBM data (the "rules" side of the comparison).

Each rule looks at one account on one calendar day and flags the transactions that
make the pattern. A rule says yes or no; it never learns. Each has an id and a
plain-English reason, and the thresholds below were set by looking at the first
60% of the timeline only (the training period), never the last 40%.

Example:
    alerts = run_rules(transactions)      # one row per (transaction, rule that fired)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd

# Structuring: amounts just under a 10,000 reporting limit (US-style limit; the data has no rupee).
REPORTING_LIMIT = 10_000
JUST_UNDER_FRACTION = 0.80  # "just under" means 80% to 99.99% of the limit
STRUCTURING_MIN_COUNT = 3  # three such payments from one account in a day is not chance

FAN_MIN_COUNTERPARTIES = 8  # distinct partners in one day; ordinary accounts rarely reach this
PASS_THROUGH_MIN_SHARE = 0.90  # money sent on in a day, as a share of money received that day
PASS_THROUGH_MIN_AMOUNT = 1_000  # ignore dust such as 0.01 test payments


@dataclass(frozen=True)
class Rule:
    """A rule id, its plain-English reason, and the function that picks the flagged rows."""

    rule_id: str
    reason: str
    find: Callable[[pd.DataFrame], pd.Series]


def _day(frame: pd.DataFrame) -> pd.Series:
    return frame["timestamp"].dt.floor("D")


def flag_structuring(frame: pd.DataFrame) -> pd.Series:
    """R1: several payments just under the reporting limit from one account in one day."""
    low = REPORTING_LIMIT * JUST_UNDER_FRACTION
    just_under = (frame["amount_paid"] >= low) & (frame["amount_paid"] < REPORTING_LIMIT)
    counts = just_under.groupby([frame["from_code"], _day(frame)]).transform("sum")
    return just_under & (counts >= STRUCTURING_MIN_COUNT)


def flag_fan_out(frame: pd.DataFrame) -> pd.Series:
    """R2: one account paying many different accounts in one day."""
    partners = frame.groupby([frame["from_code"], _day(frame)])["to_code"].transform("nunique")
    return partners >= FAN_MIN_COUNTERPARTIES


def flag_fan_in(frame: pd.DataFrame) -> pd.Series:
    """R3: many different accounts paying one account in one day."""
    partners = frame.groupby([frame["to_code"], _day(frame)])["from_code"].transform("nunique")
    return partners >= FAN_MIN_COUNTERPARTIES


def flag_pass_through(frame: pd.DataFrame) -> pd.Series:
    """R4: an account sends on nearly everything it received that day."""
    day = _day(frame)
    received = frame.groupby([frame["to_code"], day])["amount_paid"].sum()
    key = pd.MultiIndex.from_arrays([frame["from_code"], day])
    received_by_sender = pd.Series(received.reindex(key).to_numpy(), index=frame.index).fillna(0.0)
    sent = frame.groupby([frame["from_code"], day])["amount_paid"].transform("sum")
    enough = received_by_sender >= PASS_THROUGH_MIN_AMOUNT
    return enough & (sent >= PASS_THROUGH_MIN_SHARE * received_by_sender) & (sent <= received_by_sender * 1.1)


def flag_foreign_exchange_hops(frame: pd.DataFrame) -> pd.Series:
    """R5: an account sends money on in a currency it did not receive that day."""
    day = _day(frame)
    received = pd.MultiIndex.from_arrays([frame["to_code"], day, frame["receiving_currency"].astype(str)]).unique()
    received_accounts = pd.MultiIndex.from_arrays([received.get_level_values(0), received.get_level_values(1)]).unique()
    sender_day = pd.MultiIndex.from_arrays([frame["from_code"], day])
    sent_with_currency = pd.MultiIndex.from_arrays([frame["from_code"], day, frame["payment_currency"].astype(str)])
    received_something = sender_day.isin(received_accounts)
    same_currency = sent_with_currency.isin(received)
    return pd.Series(received_something & ~same_currency, index=frame.index)


RULES: List[Rule] = [
    Rule("R1", "Several payments just under the reporting limit in one day (structuring)", flag_structuring),
    Rule("R2", "One account paid many different accounts in one day (fan-out)", flag_fan_out),
    Rule("R3", "Many different accounts paid one account in one day (fan-in)", flag_fan_in),
    Rule("R4", "Nearly everything received was sent on the same day (pass-through)", flag_pass_through),
    Rule("R5", "Money was sent on in a different currency than it arrived in (FX hop)", flag_foreign_exchange_hops),
]


def run_rules(frame: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per rule id, plus `any_rule`. `frame` needs account codes (see loader)."""
    flags = pd.DataFrame(index=frame.index)
    for rule in RULES:
        flags[rule.rule_id] = rule.find(frame).astype(bool)
    flags["any_rule"] = flags[[rule.rule_id for rule in RULES]].any(axis=1)
    return flags


def score_flags(flags: pd.Series, truth: pd.Series) -> Dict[str, float]:
    """Precision, recall and count of alerts for one yes/no column against the true labels."""
    true_positive = int((flags & (truth == 1)).sum())
    alerts = int(flags.sum())
    positives = int((truth == 1).sum())
    precision = true_positive / alerts if alerts else 0.0
    recall = true_positive / positives if positives else 0.0
    return {"alerts": alerts, "precision": precision, "recall": recall}
