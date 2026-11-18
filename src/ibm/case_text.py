"""Plain-English findings, case note and draft report for an IBM alert.

Python writes every sentence from counted numbers, so no number appears that
was not counted. The wording check from the main project refuses accusing words.
Nothing here files or sends anything: the report is a draft for an analyst.

Example:
    findings = build_findings(alert_facts, evidence)
    note = draft_case_note(alert_facts, findings)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.case_note import assert_clean_wording
from src.ibm.evidence import HISTORY_DAYS, Evidence
from src.money import plural

DRAFT_NOTICE = "Draft for analyst review. Nothing has been filed or sent."
DATA_NOTICE = "Public IBM AML benchmark data (synthetic). Prototype, not certified for compliance."
TYPOLOGY_WORDS = {
    "FAN_OUT": "Fan-out",
    "FAN_IN": "Fan-in",
    "CYCLE": "Cycle",
    "BIPARTITE": "Bipartite",
    "GATHER_SCATTER": "Gather-scatter",
    "SCATTER_GATHER": "Scatter-gather",
    "STACK": "Stack",
    "RANDOM": "Random",
    "UNLISTED": "Not described by the patterns file",
    "UNKNOWN": "Not determined",
}


@dataclass(frozen=True)
class AlertFacts:
    """The alert itself, as the model flagged it."""

    alert_id: str
    txn_id: int
    time: str
    sender: str
    receiver: str
    amount: float
    currency: str
    score: float
    typology: str
    reason: str = ""  # the counted graph check behind the typology, if one fired


def typology_words(name: str) -> str:
    """'FAN_OUT' becomes 'Fan-out'."""
    return TYPOLOGY_WORDS.get(name, name)


def count_text(number: float, noun: str) -> str:
    """'1 payment' or '4 payments'."""
    return plural(int(number), noun)


def build_findings(facts: AlertFacts, evidence: Evidence) -> List[str]:
    """Every finding, as short sentences with the counted numbers."""
    n = evidence.numbers
    days = "%d days" % HISTORY_DAYS
    findings = [
        "Alert payment: %.2f %s from %s to %s at %s (model score %.3f)."
        % (facts.amount, facts.currency, facts.sender, facts.receiver, facts.time, facts.score),
        "Sender %s made %s to %s in the %s up to the alert, totalling %.2f (%s; %.0f%% crossed banks)."
        % (
            facts.sender,
            count_text(n["sender_sent_count_7d"], "payment"),
            count_text(n["sender_distinct_receivers_7d"], "different account"),
            days,
            n["sender_sent_total_7d"],
            count_text(n["sender_currencies_7d"], "currency"),
            100 * n["sender_cross_bank_share_7d"],
        ),
        "Sender velocity: %s in the last hour and %d in the last 24 hours."
        % (count_text(n["sender_sent_count_1h"], "payment"), n["sender_sent_count_24h"]),
        "Sender %s received %s from %s, totalling %.2f, in the %s."
        % (
            facts.sender,
            count_text(n["sender_received_count_7d"], "payment"),
            count_text(n["sender_distinct_payers_7d"], "different account"),
            n["sender_received_total_7d"],
            days,
        ),
        "Receiver %s received %s from %s and sent %s to %s in the %s."
        % (
            facts.receiver,
            count_text(n["receiver_received_count_7d"], "payment"),
            count_text(n["receiver_distinct_payers_7d"], "different account"),
            count_text(n["receiver_sent_count_7d"], "payment"),
            count_text(n["receiver_distinct_receivers_7d"], "account"),
            days,
        ),
    ]
    if facts.reason:
        findings.append("Pattern check: %s" % facts.reason)
    if n["receiver_paid_sender_back_7d"] > 0:
        findings.append(
            "Receiver paid the sender back %s in the %s."
            % (count_text(n["receiver_paid_sender_back_7d"], "time"), days)
        )
    return findings


def draft_case_note(facts: AlertFacts, findings: List[str], evidence: Evidence) -> str:
    """The case note as Markdown. Raises GuardrailError if the wording check fails."""
    row_ids = sorted({row["txn_id"] for row in evidence.sender_rows + evidence.receiver_rows})
    lines = [
        "# Case note: %s" % facts.alert_id,
        "",
        "**Model score:** %.3f  ·  **Suggested typology:** %s" % (facts.score, typology_words(facts.typology)),
        "",
        "## Computed findings",
    ]
    lines += ["- %s" % finding for finding in findings]
    lines += ["", "Evidence rows shown: %s" % ", ".join(str(i) for i in row_ids)]
    lines += ["", "## Suggested next step", "Refer for review by a senior analyst before any action. " + DRAFT_NOTICE]
    lines += ["", "_%s_" % DATA_NOTICE]
    note = "\n".join(lines)
    assert_clean_wording(note)
    return note


def draft_report(facts: AlertFacts, findings: List[str]) -> str:
    """A draft suspicious transaction report that names the suggested typology. Never filed automatically."""
    lines = [
        "# DRAFT suspicious transaction report (not filed)",
        "",
        "**Alert:** %s  ·  **Date:** %s" % (facts.alert_id, facts.time),
        "**Suggested typology:** %s" % typology_words(facts.typology),
        "**Accounts:** sender %s, receiver %s" % (facts.sender, facts.receiver),
        "",
        "## Grounds for referral",
        "This payment is referred for review because the counted activity resembles the pattern named above:",
    ]
    lines += ["- %s" % finding for finding in findings]
    lines += ["", "## Status", DRAFT_NOTICE, "", "_%s_" % DATA_NOTICE]
    text = "\n".join(lines)
    assert_clean_wording(text)
    return text
