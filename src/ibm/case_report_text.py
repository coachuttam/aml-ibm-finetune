"""The wording and tables of results/case_typology.md. Takes the numbers from case_report.py."""

from __future__ import annotations

from typing import Dict, List

from src.ibm.sft_format import TYPOLOGIES

LLM_PAYMENT_ACCURACY = 0.12  # fine-tuned LLM, one payment at a time: results/ibm_sft_eval.md
GRAPH_PAYMENT_ACCURACY = 0.361  # graph checks on the 576 SFT test laundering payments: results/ibm_sft_eval.md
GRAPH_ALL_TEST_ACCURACY = (
    0.40  # graph checks on all 1,339 listed test-period laundering payments: results/ibm_compare.md
)

GLOSSARY = [
    (
        "Payments, accounts, pairs",
        "How many payments are in the case, how many accounts they touch, and how many different 'A paid B' pairs.",
    ),
    (
        "Senders only / receivers only / middlemen",
        "Accounts that only send, only receive, or do both. A stack has many middlemen; a fan-out has one sender and many receivers.",
    ),
    ("Biggest fan", "The most different accounts that paid into one account, or were paid by one account."),
    ("Cycle", "Money that returns to an account it left. We record yes/no and the shortest way round (in payments)."),
    ("Depth", "How many hops the money travels before it stops. A fan is 1 deep; a stack is deep."),
    (
        "Two-sided",
        "Senders on one side, receivers on the other, nobody in the middle (the bipartite shape), and how fully the two sides are joined.",
    ),
    ("Time span", "Hours from the first payment of the case to the last."),
    ("Amounts", "Total, average and largest payment, and how much they vary relative to the average."),
    ("Cross-bank share", "Share of payments that move between two different banks."),
]


def pct(value: float) -> str:
    return "%.1f%%" % (100 * value)


def accuracy_with_range(result: Dict[str, object]) -> str:
    low, high = result["accuracy_range"]
    return "%s (95%% range %s to %s, %d cases)" % (pct(result["accuracy"]), pct(low), pct(high), result["cases"])


def per_class_table(result: Dict[str, object]) -> List[str]:
    lines = ["| Typology | Cases | Precision | Recall | F1 |", "|---|---|---|---|---|"]
    for label in TYPOLOGIES:
        s = result["per_class"][label]
        lines.append("| %s | %d | %.2f | %.2f | %.2f |" % (label, s["cases"], s["precision"], s["recall"], s["f1"]))
    return lines


def confusion_table(result: Dict[str, object]) -> List[str]:
    lines = ["| true \\ named | " + " | ".join(TYPOLOGIES) + " |", "|---|" + "---|" * len(TYPOLOGIES)]
    for truth in TYPOLOGIES:
        row = result["confusion"][truth]
        lines.append("| %s | " % truth + " | ".join(str(row[g]) for g in TYPOLOGIES) + " |")
    return lines


def flagged_section(title: str, result: Dict[str, object]) -> List[str]:
    case = result["case_level"]
    every = result["all_listed_test_payments"]
    sft = result["sft_test_payments"]
    return [
        "### %s" % title,
        "",
        "- Flagged payments: %d, grouped into %d cases (median size %d payment). %d cases contain a listed laundering "
        "payment; %d contain none (false alarms, not scored here); %d are pure (one typology only)."
        % (
            result["flagged_payments"],
            case["cases"],
            case["median_case_size"],
            case["cases_with_listed_laundering"],
            case["cases_without"],
            case["pure_cases"],
        ),
        "- **Case-level accuracy** (cases with a listed laundering payment, named by their majority typology): %s."
        % accuracy_with_range({**case, "cases": case["cases_with_listed_laundering"]}),
        "- Case-level accuracy on pure cases only: %s." % pct(case["pure_accuracy"]),
        "- Case-level accuracy by case size: "
        + "; ".join("%s: %s (%d cases)" % (k, pct(v["accuracy"]), v["cases"]) for k, v in case["by_size"].items())
        + ".",
        "- Payment-level, each flagged payment takes its case's name, payments the model did not flag count as wrong:",
        "  - on the 576 laundering payments of the fine-tuning test set: %s (%d flagged of %d); among flagged only: %s"
        % (
            pct(sft["accuracy_counting_missed_as_wrong"]),
            sft["flagged"],
            sft["payments"],
            pct(sft["accuracy_among_flagged"]),
        ),
        "  - on all %d listed laundering payments of the test period: %s; among flagged only: %s"
        % (every["payments"], pct(every["accuracy_counting_missed_as_wrong"]), pct(every["accuracy_among_flagged"])),
        "",
    ]


def write_markdown(results: Dict[str, object]) -> str:
    """The whole report as Markdown."""
    counts = results["attempt_counts"]
    test = results["oracle_test"]
    real = results["threshold_flags"]["sft_test_payments"]["accuracy_counting_missed_as_wrong"]
    lines = [
        "# Case-level typology: naming the shape of a whole case",
        "",
        "Naming a typology one payment at a time fails (fine-tuned LLM %s, graph checks %s on the same payments). "
        "A typology is a property of a group of payments, so here we name the case instead: count its shape in Python, "
        "and let a small LightGBM pick the typology from those counts. The LLM is not involved."
        % (pct(LLM_PAYMENT_ACCURACY), pct(GRAPH_PAYMENT_ACCURACY)),
        "",
        "## Headline",
        "",
        "| Setting | Accuracy |",
        "|---|---|",
        "| LLM, one payment at a time | %s |" % pct(LLM_PAYMENT_ACCURACY),
        "| Graph checks, one payment at a time | %s (%s on all 1,339 test-period payments) |"
        % (pct(GRAPH_PAYMENT_ACCURACY), pct(GRAPH_ALL_TEST_ACCURACY)),
        "| **Ceiling**: whole attempts from the patterns file, named by shape, account-disjoint test | %s |"
        % accuracy_with_range(test),
        "| **Ceiling**: same, time-only test (includes attempts that reuse earlier accounts) | %s |"
        % accuracy_with_range(results["oracle_test_time_only"]),
        "| **Real setting**: the model's flagged payments grouped into cases, scored per payment on the same 576 payments (unflagged = wrong) | %s |"
        % pct(real),
        "",
        "## What the shape numbers mean",
        "",
    ]
    lines += ["- **%s**: %s" % pair for pair in GLOSSARY]
    lines += ["", "## Step 1. Ceiling: whole attempts", ""]
    lines += [
        "Attempts are the blocks of `HI-Small_Patterns.txt`. They are split by the time of their first payment (same cut as build_sft: train earliest 60%%, validation next 20%%, test last 20%%), "
        "then any attempt that uses an account from an earlier split is dropped. Attempts per split after that rule: train %d, validation %d, test %d "
        "(by time alone: %d / %d / %d)."
        % (
            *[counts["disjoint"][k] for k in ("train", "val", "test")],
            *[counts["time_only"][k] for k in ("train", "val", "test")],
        ),
        "The classifier is trained on the %d training attempts only." % counts["disjoint"]["train"],
        "",
        "- Test accuracy, account-disjoint split: %s. Macro-F1 %.3f." % (accuracy_with_range(test), test["macro_f1"]),
        "- Test accuracy, time-only split (more cases, less strict): %s. Macro-F1 %.3f."
        % (accuracy_with_range(results["oracle_test_time_only"]), results["oracle_test_time_only"]["macro_f1"]),
        "- Validation accuracy (account-disjoint): %s." % accuracy_with_range(results["oracle_val"]),
        "",
        "### Per class (account-disjoint test)",
        "",
    ]
    lines += per_class_table(test) + ["", "### Confusion matrix (account-disjoint test)", ""] + confusion_table(test)
    lines += ["", "### Which shape numbers the classifier used most (total gain)", ""]
    lines += ["- %s: %.0f" % (name, gain) for name, gain in results["importance"][:8]]
    lines += ["", "## Step 2. Real setting: flagged payments grouped into cases", ""]
    lines += [
        "The transaction model (LightGBM, src/ibm/train.py) flags test-period payments. Flagged payments that share an account and are at most 7 days apart "
        "are joined into one case (connected components). The shape numbers are counted on the flagged payments of each case only, and the *same* classifier names it.",
        "",
    ]
    lines += flagged_section("At the model's own threshold (chosen on validation)", results["threshold_flags"])
    lines += flagged_section("Top 5,000 flagged payments (a larger alert budget)", results["budget_flags"])
    lines += [
        "## Caveats",
        "",
        "- The test sets are small (see the ranges). Treat differences of a few points as noise.",
        "- The ceiling uses the patterns file's own grouping, which a real analyst would not have. It shows what shape alone can tell.",
        "- In the real setting the classifier was trained on complete attempts but sees partial, merged cases. That mismatch is deliberate: nothing was tuned to close it.",
        "- Payment-level numbers count payments the transaction model never flagged as wrong, so recall of the flagging step caps them.",
        "- Case truth is the most common listed typology among the case's payments. Cases with no listed laundering payment are false alarms and are not scored for typology.",
        "",
    ]
    return "\n".join(lines)
