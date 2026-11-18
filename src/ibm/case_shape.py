"""Describe the shape of a case (a small group of payments) with a handful of counted numbers.

A "case" is the set of payments that belong together: one laundering attempt from the patterns
file, or a group of flagged payments that touch the same accounts. Python counts everything
here from the payments' senders, receivers, times and amounts. Nothing is read from a label.

The features, in plain words (also used on the demo slide):
  transactions / nodes / edges   how many payments, accounts, and different "who paid whom" pairs
  repeat_ratio                   payments per pair; above 1 means the same pair paid more than once
  sources / sinks / middle_nodes accounts that only send / only receive / do both
  max_in_degree / max_out_degree the most different accounts paying into, or paid by, one account
  has_cycle / shortest_cycle     money that comes back to an account it left, and the shortest way round
  layers                         how many hops deep the money travels before it stops
  bipartite_flag / _density      two groups only: senders on one side, receivers on the other
  span_hours                     time from the first payment to the last
  amount_total/mean/max/cv       how much moved; cv is spread relative to the average (0 = all equal)
  cross_bank_share               share of payments that cross from one bank to another

Example:
    rows = CaseRows(["A", "A"], ["B", "C"], [0.0, 3600.0], [10.0, 20.0], [True, False])
    shape_features(rows)["max_out_degree"]      # 2
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

SECONDS_PER_HOUR = 3600.0
MAX_CYCLE_SEARCH_NODES = 300  # a cycle search from every node is quadratic; this keeps big merged cases quick
NO_CYCLE_LENGTH = 0  # shortest_cycle when the case has no cycle

SHAPE_FEATURES: List[str] = [
    "transactions",
    "nodes",
    "edges",
    "repeat_ratio",
    "sources",
    "sinks",
    "middle_nodes",
    "max_in_degree",
    "max_out_degree",
    "has_cycle",
    "shortest_cycle",
    "layers",
    "bipartite_flag",
    "bipartite_density",
    "span_hours",
    "amount_total",
    "amount_mean",
    "amount_max",
    "amount_cv",
    "cross_bank_share",
]


@dataclass(frozen=True)
class CaseRows:
    """The payments of one case, as parallel lists."""

    senders: List[str]
    receivers: List[str]
    seconds: List[float]
    amounts: List[float]
    crosses_bank: List[bool]


Adjacency = Dict[str, Set[str]]


def build_adjacency(rows: CaseRows) -> Tuple[Adjacency, Adjacency]:
    """Who paid whom, as 'paid' and 'paid by' maps. A payment to oneself is not an edge."""
    paid: Adjacency = {}
    paid_by: Adjacency = {}
    for sender, receiver in zip(rows.senders, rows.receivers):
        if sender == receiver:
            continue
        paid.setdefault(sender, set()).add(receiver)
        paid_by.setdefault(receiver, set()).add(sender)
        paid.setdefault(receiver, set())
        paid_by.setdefault(sender, set())
    return paid, paid_by


def count_layers(paid: Adjacency, paid_by: Adjacency) -> int:
    """Hops of depth: peel off accounts nobody paid, round after round, and count the rounds minus the last."""
    waiting = {node: len(payers) for node, payers in paid_by.items()}
    layer = [node for node, count in waiting.items() if count == 0]
    rounds = 0
    while layer:
        rounds += 1
        following = []
        for node in layer:
            for receiver in paid[node]:
                waiting[receiver] -= 1
                if waiting[receiver] == 0:
                    following.append(receiver)
        layer = following
    return max(rounds - 1, 0)  # the last round only holds accounts that receive and stop, so it is not a hop


def cycle_core(paid: Adjacency, paid_by: Adjacency) -> Set[str]:
    """Accounts left after repeatedly removing any account that only sends or only receives.

    The core is empty exactly when the case has no cycle."""
    out_count = {node: len(receivers) for node, receivers in paid.items()}
    in_count = {node: len(payers) for node, payers in paid_by.items()}
    alive = set(paid)
    queue = deque(node for node in alive if out_count[node] == 0 or in_count[node] == 0)
    while queue:
        node = queue.popleft()
        if node not in alive:
            continue
        alive.discard(node)
        for receiver in paid[node]:
            in_count[receiver] -= 1
            if receiver in alive and in_count[receiver] == 0:
                queue.append(receiver)
        for payer in paid_by[node]:
            out_count[payer] -= 1
            if payer in alive and out_count[payer] == 0:
                queue.append(payer)
    return alive


def shortest_cycle_through(start: str, paid: Adjacency, core: Set[str]) -> int:
    """Fewest payments needed to leave `start` and get back to it, inside the core (0 if none)."""
    distance = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for receiver in paid[node]:
            if receiver == start:
                return distance[node] + 1
            if receiver in core and receiver not in distance:
                distance[receiver] = distance[node] + 1
                queue.append(receiver)
    return NO_CYCLE_LENGTH


def shortest_cycle(paid: Adjacency, core: Set[str]) -> int:
    """Length of the shortest cycle in the case, or 0 when there is none."""
    lengths = [shortest_cycle_through(n, paid, core) for n in sorted(core)[:MAX_CYCLE_SEARCH_NODES]]
    found = [length for length in lengths if length > 0]
    return min(found) if found else NO_CYCLE_LENGTH


def amount_features(amounts: List[float]) -> Dict[str, float]:
    """Total, mean, largest and relative spread of the amounts."""
    values = np.array(amounts, dtype=float)
    mean = float(values.mean()) if len(values) else 0.0
    spread = float(values.std() / mean) if mean > 0 else 0.0
    return {
        "amount_total": float(values.sum()),
        "amount_mean": mean,
        "amount_max": float(values.max()) if len(values) else 0.0,
        "amount_cv": spread,
    }


def bipartite_features(paid: Adjacency, paid_by: Adjacency) -> Dict[str, float]:
    """Is the case two groups only (nobody both sends and receives), and how fully do they connect?"""
    sources = [n for n in paid if not paid_by[n]]
    sinks = [n for n in paid if not paid[n]]
    two_groups_only = len(sources) + len(sinks) == len(paid)
    flag = two_groups_only and len(sources) >= 2 and len(sinks) >= 2
    edges = sum(len(receivers) for receivers in paid.values())
    density = edges / (len(sources) * len(sinks)) if flag else 0.0
    return {"bipartite_flag": float(flag), "bipartite_density": density}


def shape_features(rows: CaseRows) -> Dict[str, float]:
    """Every shape feature of one case (see SHAPE_FEATURES)."""
    paid, paid_by = build_adjacency(rows)
    core = cycle_core(paid, paid_by)
    edges = sum(len(receivers) for receivers in paid.values())
    count = len(rows.senders)
    features = {
        "transactions": float(count),
        "nodes": float(len(paid)),
        "edges": float(edges),
        "repeat_ratio": count / edges if edges else 0.0,
        "sources": float(sum(1 for n in paid if not paid_by[n])),
        "sinks": float(sum(1 for n in paid if not paid[n])),
        "middle_nodes": float(sum(1 for n in paid if paid[n] and paid_by[n])),
        "max_in_degree": float(max((len(v) for v in paid_by.values()), default=0)),
        "max_out_degree": float(max((len(v) for v in paid.values()), default=0)),
        "has_cycle": float(bool(core)),
        "shortest_cycle": float(shortest_cycle(paid, core)),
        "layers": float(count_layers(paid, paid_by)),
        "span_hours": (max(rows.seconds) - min(rows.seconds)) / SECONDS_PER_HOUR if count else 0.0,
        "cross_bank_share": float(np.mean(rows.crosses_bank)) if count else 0.0,
    }
    features.update(bipartite_features(paid, paid_by))
    features.update(amount_features(rows.amounts))
    return features
