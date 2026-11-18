"""The text a model reads and writes for IBM typology fine-tuning.

Training, evaluation and any later use import this one file, so the prompt can
never drift apart. Python writes the input and the target note from counted
numbers (evidence.py), so every number in the note is already in the input.
The model only has to choose the typology and phrase the note.

Example:
    text = build_input_text(facts, evidence)
    answer = target_answer("FAN_OUT", evidence)   # '{"typology": "FAN_OUT", "note": "..."}'
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from src.ibm.case_text import AlertFacts, build_findings
from src.ibm.evidence import HISTORY_DAYS, Evidence
from src.ibm.loader import NOT_LAUNDERING
from src.money import plural
from src.schema import BANNED_WORDS, numbers_in, parse_json_object

TYPOLOGIES = (
    "FAN_OUT",
    "FAN_IN",
    "CYCLE",
    "BIPARTITE",
    "GATHER_SCATTER",
    "SCATTER_GATHER",
    "STACK",
    "RANDOM",
)
NO_PATTERN = NOT_LAUNDERING  # "NONE": nothing in the counted activity matches a listed pattern
IBM_LABELS = TYPOLOGIES + (NO_PATTERN,)
ROWS_SHOWN_PER_SIDE = 3  # Qwen reads digits one by one, so every row is costly; 3 keeps inputs near 800 tokens
MINUTE_LENGTH = 16  # 'YYYY-MM-DD HH:MM'; the seconds are always 00 in this data and cost tokens
MAX_NOTE_LENGTH = 400  # characters; the note is two short sentences
CLEAN_WORDING_NOTICE = "Refer for review by an analyst."

MEANINGS: Dict[str, str] = {
    "FAN_OUT": "Looks like fan-out: one account paying many different accounts.",
    "FAN_IN": "Looks like fan-in: many different accounts paying one account.",
    "CYCLE": "Looks like a cycle: money that returns to where it started.",
    "BIPARTITE": "Looks like a bipartite pattern: one group of accounts paying another group.",
    "GATHER_SCATTER": "Looks like gather-scatter: an account takes in from many, then pays out to many.",
    "SCATTER_GATHER": "Looks like scatter-gather: money spread across accounts, then gathered into one.",
    "STACK": "Looks like a stack: money passed through layers of accounts.",
    "RANDOM": "Looks like a random pattern: money moved between accounts with no clear shape.",
    NO_PATTERN: "No listed laundering pattern is visible in the counted activity.",
}

SYSTEM_PROMPT = (
    "You are an anti-money-laundering assistant. Read the alert and its evidence, then reply "
    'with JSON only, in this form: {"typology": ..., "note": ...}. '
    "typology is one of: %s. "
    % ", ".join(IBM_LABELS)
    + "note is two short sentences that use only numbers found in the alert. "
    "Say 'refer for review', never accuse anyone."
)


def evidence_row_text(row: dict) -> str:
    """One transaction on one line, without its internal id."""
    return "%s  %s -> %s  %.2f %s  %s" % (
        row["time"][:MINUTE_LENGTH],
        row["from"],
        row["to"],
        row["amount"],
        row["currency"],
        row["format"],
    )


def build_input_text(facts: AlertFacts, evidence: Evidence) -> str:
    """The alert and its counted evidence, in the order investigate.py shows them.

    The model score is left out (there is no model score in this task) and so is the
    graph check's reason, which would hand the model the baseline's answer.
    """
    # findings[0] is the alert line with a model score; it is replaced by the same line without one.
    alert_line = "Alert payment: %.2f %s from %s to %s at %s." % (
        facts.amount,
        facts.currency,
        facts.sender,
        facts.receiver,
        facts.time,
    )
    lines: List[str] = [alert_line] + build_findings(facts, evidence)[1:]
    lines += ["", "Latest payments sent by the sender:"]
    lines += [evidence_row_text(r) for r in evidence.sender_rows[-ROWS_SHOWN_PER_SIDE:]]
    lines += ["", "Latest payments received by the receiver:"]
    lines += [evidence_row_text(r) for r in evidence.receiver_rows[-ROWS_SHOWN_PER_SIDE:]]
    return "\n".join(lines)


def counted_sentence(evidence: Evidence) -> str:
    """One sentence of counted facts, worded exactly as in the input findings."""
    n = evidence.numbers
    return "In the %d days up to the alert the sender made %s to %s and the receiver received %s from %s." % (
        HISTORY_DAYS,
        plural(int(n["sender_sent_count_7d"]), "payment"),
        plural(int(n["sender_distinct_receivers_7d"]), "different account"),
        plural(int(n["receiver_received_count_7d"]), "payment"),
        plural(int(n["receiver_distinct_payers_7d"]), "different account"),
    )


def target_note(typology: str, evidence: Evidence) -> str:
    """What the model should write: the typology's plain meaning, the counted facts, then the referral line."""
    return "%s %s %s" % (
        MEANINGS[typology],
        counted_sentence(evidence),
        CLEAN_WORDING_NOTICE,
    )


def target_answer(typology: str, evidence: Evidence) -> str:
    """The exact JSON text the model is trained to produce."""
    return json.dumps(
        {"typology": typology, "note": target_note(typology, evidence)},
        ensure_ascii=False,
    )


def build_messages(input_text: str) -> List[Dict[str, str]]:
    """The conversation up to the point where the model should answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": input_text},
    ]


def find_answer_problems(answer: dict, input_text: str) -> List[str]:
    """Everything wrong with a parsed answer. An empty list means it is fine."""
    problems: List[str] = []
    if answer.get("typology") not in IBM_LABELS:
        problems.append("typology is not one of the allowed labels")
    note = answer.get("note")
    if not isinstance(note, str) or not note.strip():
        return problems + ["note is missing"]
    if len(note) > MAX_NOTE_LENGTH:
        problems.append("note is too long")
    if any(word in note.lower() for word in BANNED_WORDS):
        problems.append("note uses accusing wording")
    if numbers_in(note) - numbers_in(input_text):
        problems.append("note uses a number that is not in the input")
    return problems


def typology_of_answer(raw_text: str, input_text: str) -> Optional[Dict[str, object]]:
    """Parse and check one raw model answer. Returns a plain dict, or None if it is not JSON."""
    answer = parse_json_object(raw_text)
    if answer is None:
        return None
    problems = find_answer_problems(answer, input_text)
    return {
        "typology": answer.get("typology"),
        "note": answer.get("note"),
        "problems": problems,
    }
