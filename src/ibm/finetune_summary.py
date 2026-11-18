"""Base model versus fine-tuned model on the original (rupee) alerts, from the saved answers.

Reads results/base_<split>.jsonl and results/finetuned_<split>.jsonl and scores them
with the same functions as `python -m src.evaluate`. No model is run and nothing is
tuned. The held-out numbers come from answers saved earlier; they are only re-read.

Example:
    python -m src.ibm.finetune_summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from src.dataset import load_split
from src.evaluate import DATA_FOLDER, RESULTS_FOLDER, evaluate_predictions, load_predictions, model_extras, typology_of
from src.ibm.report_extras import markdown_table

SPLITS = ("validation", "test", "heldout")
SYSTEMS = {"Base model": "base", "Fine-tuned": "finetuned"}
SUMMARY_PATH = RESULTS_FOLDER / "finetune_summary.md"


def accuracy(examples: List[dict], predicted: List[str]) -> float:
    """Share of alerts whose typology the model got exactly right."""
    right = sum(1 for example, guess in zip(examples, predicted) if example["label"]["typology"] == guess)
    return right / len(examples) if examples else 0.0


def score_system(split: str, prefix: str, results_folder: Path = RESULTS_FOLDER) -> Optional[Dict[str, float]]:
    """Accuracy, macro-F1 and the model checks for one system on one split, or None if no answers file."""
    path = results_folder / ("%s_%s.jsonl" % (prefix, split))
    if not path.exists():
        return None
    examples = load_split(DATA_FOLDER / ("%s.jsonl" % split))
    answers = load_predictions(path)
    predicted = [typology_of(answers.get(example["alert_id"])) for example in examples]
    scores = evaluate_predictions(examples, predicted)
    extras = model_extras(examples, answers)
    return {"accuracy": accuracy(examples, predicted), "macro_f1": scores["macro_f1"], **extras}


def table_rows(results_folder: Path = RESULTS_FOLDER) -> List[List[str]]:
    """One row per split and system."""
    rows = []
    for split in SPLITS:
        for name, prefix in SYSTEMS.items():
            score = score_system(split, prefix, results_folder)
            label = split if split != "heldout" else "held-out (saved answers, not re-run)"
            if score is None:
                rows.append([label, name, "no answers", "", "", "", ""])
                continue
            rows.append(
                [
                    label,
                    name,
                    "%.3f" % score["accuracy"],
                    "%.3f" % score["macro_f1"],
                    "%.1f%%" % (100 * score["valid_answers"]),
                    "%.1f%%" % (100 * score["faithful_reasons"]),
                    "%.1f%%" % (100 * score["risk_accuracy"]),
                ]
            )
    return rows


def build_summary(results_folder: Path = RESULTS_FOLDER) -> str:
    """The markdown summary."""
    header = ["Split", "System", "Accuracy", "Macro-F1", "Valid answers", "Faithful reasons", "Risk accuracy"]
    return (
        "\n\n".join(
            [
                "# Base versus fine-tuned model (original alerts)",
                "Typology accuracy and macro-F1 over the five labels, scored by the same functions as "
                "`python -m src.evaluate`, from the saved answers in `results/`.",
                markdown_table(header, table_rows(results_folder)),
            ]
        )
        + "\n"
    )


def main() -> None:
    text = build_summary()
    SUMMARY_PATH.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
