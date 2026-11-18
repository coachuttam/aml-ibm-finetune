"""Name a laundering typology by checking the shape of the 7-day neighbourhood around an alert.

Python walks the payment graph (who paid whom in the 7 days up to the alert) and
looks for the classic shapes. The checks run in a fixed order, most specific first:
cycle, scatter-gather, bipartite, plain fan-out or fan-in, then gather-scatter last (it fires
on almost any busy account, so it only wins when nothing simpler fits; order chosen on training alerts).
When no shape fits (the stack and random patterns have none we can check), the
caller falls back to the learned classifier in typology.py.

Example:
    verdict = classify_by_graph(index, sender, receiver, end, GraphConfig())
    verdict.typology, verdict.reason     # ("FAN_OUT", "Sender paid 9 different accounts ...")
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.ibm.evidence import HISTORY_SECONDS, TransactionIndex

MAX_ROWS_PER_ACCOUNT = 200  # a hub can have thousands of rows; the latest ones are enough to see a shape
MAX_NODES_VISITED = 300  # stops a path search from wandering across the whole graph
MAX_ACCOUNTS_EXPANDED = 30  # accounts whose own neighbours we look at, for the two-step shapes
BIPARTITE_MIN_SIDE = 2  # at least two accounts on each side before we call it bipartite
DEGREE_CHOICES = (2, 3, 5, 8)  # candidate values of k; the best one is chosen on the training period
HOP_CHOICES = (3, 5)  # candidate longest cycles
MAX_TUNING_ROWS = 300  # training alerts used to choose k; enough for a stable pick, quick to run


@dataclass(frozen=True)
class GraphConfig:
    """The two settings: how many neighbours make a fan, and how long a cycle may be."""

    degree_min: int = 5
    max_cycle_hops: int = 5


@dataclass(frozen=True)
class GraphVerdict:
    """A typology and the counted reason for it."""

    typology: str
    reason: str


class Neighbourhood:
    """Who paid whom in the 7 days up to one moment, with each lookup remembered."""

    def __init__(self, index: TransactionIndex, end: int) -> None:
        self.index, self.end = index, end
        self.out_memo: Dict[int, np.ndarray] = {}
        self.in_memo: Dict[int, np.ndarray] = {}

    def paid(self, account: int) -> np.ndarray:
        """Different accounts this account paid in the window."""
        if account not in self.out_memo:
            rows = self.index.window(self.index.sent_by, account, self.end, HISTORY_SECONDS)[-MAX_ROWS_PER_ACCOUNT:]
            self.out_memo[account] = np.unique(self.index.subset["to_code"].to_numpy()[rows])
        return self.out_memo[account]

    def payers(self, account: int) -> np.ndarray:
        """Different accounts that paid this account in the window."""
        if account not in self.in_memo:
            rows = self.index.window(self.index.received_by, account, self.end, HISTORY_SECONDS)[-MAX_ROWS_PER_ACCOUNT:]
            self.in_memo[account] = np.unique(self.index.subset["from_code"].to_numpy()[rows])
        return self.in_memo[account]


def hops_back(hood: Neighbourhood, start: int, goal: int, max_hops: int) -> Optional[int]:
    """Fewest payments needed to get from `start` to `goal`, or None if more than `max_hops`."""
    seen = {start}
    queue = deque([(start, 0)])
    while queue and len(seen) < MAX_NODES_VISITED:
        account, hops = queue.popleft()
        if hops >= max_hops:
            continue
        for neighbour in hood.paid(account):
            if neighbour == goal:
                return hops + 1
            if neighbour not in seen:
                seen.add(int(neighbour))
                queue.append((int(neighbour), hops + 1))
    return None


def most_shared_neighbour(accounts: np.ndarray, neighbours_of, exclude: int) -> int:
    """How many of `accounts` share the most common neighbour (not counting `exclude`)."""
    counts: Counter = Counter()
    for account in accounts[:MAX_ACCOUNTS_EXPANDED]:
        counts.update(int(n) for n in neighbours_of(int(account)) if n != exclude)
    return max(counts.values()) if counts else 0


def check_cycle(hood, sender, receiver, config) -> Optional[GraphVerdict]:
    hops = hops_back(hood, receiver, sender, config.max_cycle_hops)
    if hops is None:
        return None
    return GraphVerdict("CYCLE", "Money paid to the receiver reaches the sender again after %d payments." % hops)


def check_scatter_gather(hood, sender, receiver, config) -> Optional[GraphVerdict]:
    k = config.degree_min
    paid = hood.paid(sender)
    if len(paid) >= k and most_shared_neighbour(paid, hood.paid, sender) >= k:
        return GraphVerdict(
            "SCATTER_GATHER",
            "The sender paid %d accounts and %d of them paid one common account."
            % (len(paid), most_shared_neighbour(paid, hood.paid, sender)),
        )
    payers = hood.payers(receiver)
    shared = most_shared_neighbour(payers, hood.payers, receiver) if len(payers) >= k else 0
    if shared >= k:
        reason = "%d accounts paid the receiver and %d of them were paid by one common account."
        return GraphVerdict("SCATTER_GATHER", reason % (len(payers), shared))
    return None


def check_gather_scatter(hood, sender, receiver, config) -> Optional[GraphVerdict]:
    k = config.degree_min
    for account, role in ((sender, "Sender"), (receiver, "Receiver")):
        gathered, scattered = len(hood.payers(account)), len(hood.paid(account))
        if gathered >= k and scattered >= k:
            return GraphVerdict(
                "GATHER_SCATTER", "%s was paid by %d accounts and then paid %d accounts." % (role, gathered, scattered)
            )
    return None


def check_bipartite(hood, sender, receiver, config) -> Optional[GraphVerdict]:
    targets = set(int(a) for a in hood.paid(sender))
    payers = [int(p) for p in hood.payers(receiver)[:MAX_ACCOUNTS_EXPANDED]]
    sharing = [p for p in payers if len(targets & set(int(a) for a in hood.paid(p))) >= BIPARTITE_MIN_SIDE]
    if len(targets) >= BIPARTITE_MIN_SIDE and len(sharing) >= BIPARTITE_MIN_SIDE:
        return GraphVerdict(
            "BIPARTITE",
            "%d payers each paid at least %d of the same %d accounts."
            % (len(sharing), BIPARTITE_MIN_SIDE, len(targets)),
        )
    return None


def check_fan(hood, sender, receiver, config) -> Optional[GraphVerdict]:
    """A fan is one-sided: the busy account only pays out (fan-out) or only takes in (fan-in)."""
    k = config.degree_min
    out_degree, sender_gathered = len(hood.paid(sender)), len(hood.payers(sender))
    in_degree, receiver_scattered = len(hood.payers(receiver)), len(hood.paid(receiver))
    if out_degree >= k and sender_gathered < k and out_degree >= in_degree:
        return GraphVerdict("FAN_OUT", "The sender paid %d different accounts in 7 days." % out_degree)
    if in_degree >= k and receiver_scattered < k:
        return GraphVerdict("FAN_IN", "%d different accounts paid the receiver in 7 days." % in_degree)
    return None


CHECKS = (check_cycle, check_scatter_gather, check_bipartite, check_fan, check_gather_scatter)


def classify_by_graph(index: TransactionIndex, txn_id: int, config: GraphConfig) -> Optional[GraphVerdict]:
    """The first shape that fits the alert's neighbourhood, or None if no check fires."""
    position = index.position_of(txn_id)
    row = index.subset.iloc[position]
    sender, receiver = int(row["from_code"]), int(row["to_code"])
    hood = Neighbourhood(index, int(index.seconds[position]))
    for check in CHECKS:
        verdict = check(hood, sender, receiver, config)
        if verdict is not None:
            return verdict
    return None


def graph_accuracy(
    index: TransactionIndex, txn_ids: List[int], truths: List[str], fallback: List[str], config: GraphConfig
) -> float:
    """Share of alerts typed correctly when the graph checks run first and `fallback` fills the gaps."""
    right = 0
    for txn_id, truth, backup in zip(txn_ids, truths, fallback):
        verdict = classify_by_graph(index, txn_id, config)
        right += (verdict.typology if verdict else backup) == truth
    return right / len(txn_ids) if txn_ids else 0.0


def choose_config(index: TransactionIndex, txn_ids: List[int], truths: List[str], fallback: List[str]) -> GraphConfig:
    """Pick k and the cycle length with the best accuracy on training-period alerts only."""
    step = max(len(txn_ids) // MAX_TUNING_ROWS, 1)
    sample = slice(0, None, step)
    best, best_score = GraphConfig(), -1.0
    for degree in DEGREE_CHOICES:
        for hops in HOP_CHOICES:
            config = GraphConfig(degree, hops)
            score = graph_accuracy(index, txn_ids[sample], truths[sample], fallback[sample], config)
            if score > best_score:
                best, best_score = config, score
    return best
