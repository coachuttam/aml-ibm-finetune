"""Score base model, fine-tuned model and the graph-check baseline on the SFT test split.

    python -m src.ibm.eval_sft generate --adapter PATH --output results/ibm_sft_finetuned_test.jsonl   (GPU)
    python -m src.ibm.eval_sft generate --output results/ibm_sft_base_test.jsonl                       (GPU, no adapter)
    python -m src.ibm.eval_sft score                                                                    (laptop)

`score` reads the saved answers, so no GPU is needed. It reuses `load_predictions`, `share` and
`INVALID` from src/evaluate.py and `score_label` from src/metrics.py. The faithfulness check is the one
from src/schema.py (every number in the note must be in the input), applied in sft_format.py.
Only the test split is read; train and val are never scored here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.evaluate import INVALID, load_predictions, share
from src.ibm.build_sft import load_examples
from src.ibm.sft_format import IBM_LABELS, NO_PATTERN, TYPOLOGIES, build_messages, typology_of_answer
from src.metrics import score_label
from src.predict import BASE_MODEL, load_model

RESULTS_FOLDER = Path("results")
SYSTEMS = {
    "Base model": "ibm_sft_base_test.jsonl",
    "Fine-tuned": "ibm_sft_finetuned_test.jsonl",
    "Graph-check baseline": "ibm_sft_graph_test.jsonl",
}
REPORT_PATH = RESULTS_FOLDER / "ibm_sft_eval.md"
REFERENCE_GRAPH_ACCURACY = 0.40  # 542 of 1,339 laundering rows, results/ibm_compare.md (all test-period rows)
UNFAITHFUL_PROBLEM = "note uses a number that is not in the input"
BATCH_SIZE = 16  # inputs are about 800 tokens; 16 at once fits an L4 (24 GB) in 4-bit
MAX_NEW_TOKENS = 160  # the answer is about 90 tokens


def predicted_typology(row: Optional[dict]) -> str:
    """The typology an answer gave, or INVALID when it gave none usable."""
    if row and row.get("typology") in IBM_LABELS:
        return row["typology"]
    return INVALID


def accuracy(truths: List[str], guesses: List[str]) -> float:
    """Share of guesses that equal the truth."""
    return share(sum(1 for t, g in zip(truths, guesses) if t == g), len(truths))


def macro_f1_over(truths: List[str], guesses: List[str], labels) -> float:
    """Average F1 over `labels`. INVALID never matches a label, so it counts as a miss."""
    return sum(score_label(truths, guesses, label).f1 for label in labels) / len(labels)


def score_system(examples: List[dict], answers: Dict[str, dict]) -> Dict[str, float]:
    """Accuracy, macro-F1 and faithfulness for one system."""
    truths = [e["typology"] for e in examples]
    guesses = [predicted_typology(answers.get(e["alert_id"])) for e in examples]
    pairs = [(t, g) for t, g in zip(truths, guesses) if t != NO_PATTERN]
    rows = [answers.get(e["alert_id"]) for e in examples]
    unfaithful = sum(1 for r in rows if r and UNFAITHFUL_PROBLEM in r.get("problems", []))
    return {
        "accuracy_all": accuracy(truths, guesses),
        "macro_f1_all": macro_f1_over(truths, guesses, IBM_LABELS),
        "accuracy_laundering": accuracy([t for t, _ in pairs], [g for _, g in pairs]),
        "macro_f1_laundering": macro_f1_over([t for t, _ in pairs], [g for _, g in pairs], TYPOLOGIES),
        "valid_answers": share(sum(1 for r in rows if r and r.get("valid")), len(examples)),
        "faithful_notes": 1 - share(unfaithful, len(examples)),
    }


def format_table(results: Dict[str, Dict[str, float]]) -> str:
    """A Markdown table, one row per system."""
    columns = [
        "System",
        "Accuracy (all 9 classes)",
        "Macro-F1 (all)",
        "Accuracy (laundering rows only)",
        "Macro-F1 (8 typologies)",
        "Valid answers",
        "Faithful notes",
    ]
    header = "| " + " | ".join(columns) + " |"
    lines = [header, "|---|---|---|---|---|---|---|"]
    for name, s in results.items():
        has_notes = name != "Graph-check baseline"  # the baseline writes no note, and never answers NONE
        lines.append(
            "| %s | %s | %s | %.1f%% | %.3f | %s | %s |"
            % (
                name,
                "%.1f%%" % (100 * s["accuracy_all"]) if has_notes else "n/a",
                "%.3f" % s["macro_f1_all"] if has_notes else "n/a",
                100 * s["accuracy_laundering"],
                s["macro_f1_laundering"],
                "%.1f%%" % (100 * s["valid_answers"]) if has_notes else "n/a",
                "%.1f%%" % (100 * s["faithful_notes"]) if has_notes else "n/a",
            )
        )
    return "\n".join(lines)


def score(results_folder: Path = RESULTS_FOLDER) -> str:
    """Score every system that has a saved answers file; returns the report text."""
    examples = load_examples("test")
    results: Dict[str, Dict[str, float]] = {}
    for name, filename in SYSTEMS.items():
        path = results_folder / filename
        if path.exists():
            results[name] = score_system(examples, load_predictions(path))
        else:
            print("No answers yet for %s (%s)" % (name, path))
    note = (
        "Test split: %d examples (%d laundering, %d matched non-laundering). "
        "The graph-check baseline never answers NONE, so compare systems on the laundering-only columns. "
        "The quoted %.0f%% baseline was measured on all 1,339 laundering rows of the test period; "
        "the graph-check row here is the same baseline on this smaller, balanced, account-disjoint test set."
        % (
            len(examples),
            sum(1 for e in examples if e["typology"] != NO_PATTERN),
            sum(1 for e in examples if e["typology"] == NO_PATTERN),
            100 * REFERENCE_GRAPH_ACCURACY,
        )
    )
    return "# IBM typology fine-tuning: test results\n\n%s\n\n%s\n" % (note, format_table(results))


def generate(adapter: Optional[str], output: Path, limit: Optional[int]) -> None:
    """Run a model over the test inputs (GPU). An invalid answer is retried once on its own."""
    examples = load_examples("test")[:limit]
    from src.ibm.finetune_ibm import pick_precision

    _, generate_batch = load_model(adapter, BASE_MODEL, messages_for=build_messages, compute_dtype=pick_precision()[0])
    rows = []
    for start in range(0, len(examples), BATCH_SIZE):
        chunk = examples[start : start + BATCH_SIZE]
        texts = generate_batch([e["input"] for e in chunk])
        rows += [checked_row(e, text, generate_batch) for e, text in zip(chunk, texts)]
        print("  answered %d of %d" % (len(rows), len(examples)), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print("Wrote %d answers to %s" % (len(rows), output))


def checked_row(example: dict, text: str, generate_batch) -> dict:
    """The saved form of one answer, after one retry if the first one failed the checks."""
    checked = typology_of_answer(text, example["input"])
    if checked is None or checked["problems"]:
        checked = typology_of_answer(generate_batch([example["input"]])[0], example["input"])
    if checked is None:
        return {
            "alert_id": example["alert_id"],
            "valid": False,
            "typology": None,
            "note": None,
            "problems": ["not JSON"],
        }
    return {"alert_id": example["alert_id"], "valid": not checked["problems"], **checked}


def main() -> None:
    """`generate` on a GPU, `score` anywhere."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--adapter", default=None, help="adapter folder; omit for the plain base model")
    gen.add_argument("--output", required=True)
    gen.add_argument("--limit", type=int, default=None)
    commands.add_parser("score")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.adapter, Path(args.output), args.limit)
    else:
        report = score()
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
