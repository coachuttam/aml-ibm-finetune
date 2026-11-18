"""Run the graph-check baseline on the SFT test examples, so it is scored on the very same rows as the models.

The baseline is the one in src/ibm/investigate.py: graph checks first, a LightGBM classifier as the
fallback, both tuned on training-period laundering rows only. It never answers NONE, so it can
only be compared on laundering rows.

Example:
    python -m src.ibm.graph_baseline_on_sft     # writes results/ibm_sft_graph_test.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.ibm.build_sft import load_examples
from src.ibm.evidence import TransactionIndex, subset_for_alerts
from src.ibm.graph_typology import choose_config
from src.ibm.investigate import UNKNOWN_TYPOLOGY, collect_evidence, combine_guesses, listed_laundering_positions
from src.ibm.loader import add_account_codes, load_labelled_transactions
from src.ibm.train import temporal_split
from src.ibm.typology import out_of_fold_typologies, predict_typologies, train_typology_classifier

RESULTS_PATH = Path("results/ibm_sft_graph_test.jsonl")


def transaction_number(alert_id: str) -> int:
    """'IBM-123' becomes 123 (the row's position in time order)."""
    return int(alert_id.split("-")[1])


def graph_baseline_guesses(examples: List[dict]) -> Dict[str, str]:
    """The baseline's typology for each example, keyed by alert id."""
    frame = add_account_codes(load_labelled_transactions())
    train_period, _, _ = temporal_split(len(frame))
    train_positions = listed_laundering_positions(frame, train_period)
    test_ids = [transaction_number(example["alert_id"]) for example in examples]
    index = TransactionIndex(subset_for_alerts(frame, np.concatenate([train_positions, np.array(test_ids)])))
    train_ids = [int(frame.index[p]) for p in train_positions]
    evidence = collect_evidence(frame, np.concatenate([train_positions, np.array(test_ids)]), index)
    truths = [frame.at[i, "typology"] for i in train_ids]
    classifier = train_typology_classifier([evidence[i] for i in train_ids], truths)
    fallback = out_of_fold_typologies([evidence[i] for i in train_ids], truths)
    config = choose_config(index, train_ids, truths, fallback)
    backups = predict_typologies(classifier, [evidence[i] for i in test_ids], UNKNOWN_TYPOLOGY)
    combined = combine_guesses(index, test_ids, config, backups)
    return {example["alert_id"]: guess[0] for example, guess in zip(examples, combined)}


def main() -> None:
    """Score-ready baseline answers for the test split."""
    examples = load_examples("test")
    guesses = graph_baseline_guesses(examples)
    lines = [json.dumps({"alert_id": a, "valid": True, "typology": t, "problems": []}) for a, t in guesses.items()]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote %d baseline answers to %s" % (len(lines), RESULTS_PATH))


if __name__ == "__main__":
    main()
