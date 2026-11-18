# Case-level typology: naming the shape of a whole case

Naming a typology one payment at a time fails (fine-tuned LLM 12.0%, graph checks 36.1% on the same payments). A typology is a property of a group of payments, so here we name the case instead: count its shape in Python, and let a small LightGBM pick the typology from those counts. The LLM is not involved.

## Headline

| Setting | Accuracy |
|---|---|
| LLM, one payment at a time | 12.0% |
| Graph checks, one payment at a time | 36.1% (40.0% on all 1,339 test-period payments) |
| **Ceiling**: whole attempts from the patterns file, named by shape, account-disjoint test | 97.9% (95% range 88.9% to 99.6%, 47 cases) |
| **Ceiling**: same, time-only test (includes attempts that reuse earlier accounts) | 76.8% (95% range 67.4% to 84.2%, 95 cases) |
| **Real setting**: the model's flagged payments grouped into cases, scored per payment on the same 576 payments (unflagged = wrong) | 44.6% |

## What the shape numbers mean

- **Payments, accounts, pairs**: How many payments are in the case, how many accounts they touch, and how many different 'A paid B' pairs.
- **Senders only / receivers only / middlemen**: Accounts that only send, only receive, or do both. A stack has many middlemen; a fan-out has one sender and many receivers.
- **Biggest fan**: The most different accounts that paid into one account, or were paid by one account.
- **Cycle**: Money that returns to an account it left. We record yes/no and the shortest way round (in payments).
- **Depth**: How many hops the money travels before it stops. A fan is 1 deep; a stack is deep.
- **Two-sided**: Senders on one side, receivers on the other, nobody in the middle (the bipartite shape), and how fully the two sides are joined.
- **Time span**: Hours from the first payment of the case to the last.
- **Amounts**: Total, average and largest payment, and how much they vary relative to the average.
- **Cross-bank share**: Share of payments that move between two different banks.

## Step 1. Ceiling: whole attempts

Attempts are the blocks of `HI-Small_Patterns.txt`. They are split by the time of their first payment (same cut as build_sft: train earliest 60%, validation next 20%, test last 20%), then any attempt that uses an account from an earlier split is dropped. Attempts per split after that rule: train 199, validation 44, test 47 (by time alone: 199 / 76 / 95).
The classifier is trained on the 199 training attempts only.

- Test accuracy, account-disjoint split: 97.9% (95% range 88.9% to 99.6%, 47 cases). Macro-F1 0.967.
- Test accuracy, time-only split (more cases, less strict): 76.8% (95% range 67.4% to 84.2%, 95 cases). Macro-F1 0.763.
- Validation accuracy (account-disjoint): 95.5% (95% range 84.9% to 98.7%, 44 cases).

### Per class (account-disjoint test)

| Typology | Cases | Precision | Recall | F1 |
|---|---|---|---|---|
| FAN_OUT | 5 | 1.00 | 1.00 | 1.00 |
| FAN_IN | 6 | 1.00 | 1.00 | 1.00 |
| CYCLE | 7 | 1.00 | 1.00 | 1.00 |
| BIPARTITE | 5 | 1.00 | 1.00 | 1.00 |
| GATHER_SCATTER | 9 | 1.00 | 1.00 | 1.00 |
| SCATTER_GATHER | 5 | 1.00 | 1.00 | 1.00 |
| STACK | 3 | 1.00 | 0.67 | 0.80 |
| RANDOM | 7 | 0.88 | 1.00 | 0.93 |

### Confusion matrix (account-disjoint test)

| true \ named | FAN_OUT | FAN_IN | CYCLE | BIPARTITE | GATHER_SCATTER | SCATTER_GATHER | STACK | RANDOM |
|---|---|---|---|---|---|---|---|---|
| FAN_OUT | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| FAN_IN | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| CYCLE | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 |
| BIPARTITE | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| GATHER_SCATTER | 0 | 0 | 0 | 0 | 9 | 0 | 0 | 0 |
| SCATTER_GATHER | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 |
| STACK | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 |
| RANDOM | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |

### Which shape numbers the classifier used most (total gain)

- middle_nodes: 1548
- span_hours: 1227
- sources: 1090
- max_out_degree: 749
- bipartite_flag: 688
- max_in_degree: 540
- layers: 539
- transactions: 386

## Step 2. Real setting: flagged payments grouped into cases

The transaction model (LightGBM, src/ibm/train.py) flags test-period payments. Flagged payments that share an account and are at most 7 days apart are joined into one case (connected components). The shape numbers are counted on the flagged payments of each case only, and the *same* classifier names it.

### At the model's own threshold (chosen on validation)

- Flagged payments: 1655, grouped into 889 cases (median size 1 payment). 219 cases contain a listed laundering payment; 670 contain none (false alarms, not scored here); 189 are pure (one typology only).
- **Case-level accuracy** (cases with a listed laundering payment, named by their majority typology): 36.1% (95% range 30.0% to 42.6%, 219 cases).
- Case-level accuracy on pure cases only: 34.4%.
- Case-level accuracy by case size: 1 payment: 16.8% (113 cases); 2 to 4 payments: 42.9% (49 cases); 5 or more payments: 68.4% (57 cases).
- Payment-level, each flagged payment takes its case's name, payments the model did not flag count as wrong:
  - on the 576 laundering payments of the fine-tuning test set: 44.6% (375 flagged of 576); among flagged only: 68.5%
  - on all 1339 listed laundering payments of the test period: 40.4%; among flagged only: 61.9%

### Top 5,000 flagged payments (a larger alert budget)

- Flagged payments: 5000, grouped into 3570 cases (median size 1 payment). 248 cases contain a listed laundering payment; 3322 contain none (false alarms, not scored here); 181 are pure (one typology only).
- **Case-level accuracy** (cases with a listed laundering payment, named by their majority typology): 35.1% (95% range 29.4% to 41.2%, 248 cases).
- Case-level accuracy on pure cases only: 33.7%.
- Case-level accuracy by case size: 1 payment: 19.3% (109 cases); 2 to 4 payments: 40.8% (71 cases); 5 or more payments: 54.4% (68 cases).
- Payment-level, each flagged payment takes its case's name, payments the model did not flag count as wrong:
  - on the 576 laundering payments of the fine-tuning test set: 50.0% (459 flagged of 576); among flagged only: 62.7%
  - on all 1339 listed laundering payments of the test period: 44.5%; among flagged only: 56.4%

## Caveats

- The test sets are small (see the ranges). Treat differences of a few points as noise.
- The ceiling uses the patterns file's own grouping, which a real analyst would not have. It shows what shape alone can tell.
- In the real setting the classifier was trained on complete attempts but sees partial, merged cases. That mismatch is deliberate: nothing was tuned to close it.
- Payment-level numbers count payments the transaction model never flagged as wrong, so recall of the flagging step caps them.
- Case truth is the most common listed typology among the case's payments. Cases with no listed laundering payment are false alarms and are not scored for typology.
