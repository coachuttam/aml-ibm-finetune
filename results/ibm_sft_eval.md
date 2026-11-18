# IBM typology fine-tuning: test results

Test split: 656 examples (576 laundering, 80 matched non-laundering). The graph-check baseline never answers NONE, so compare systems on the laundering-only columns. The quoted 40% baseline was measured on all 1,339 laundering rows of the test period; the graph-check row here is the same baseline on this smaller, balanced, account-disjoint test set.

| System | Accuracy (all 9 classes) | Macro-F1 (all) | Accuracy (laundering rows only) | Macro-F1 (8 typologies) | Valid answers | Faithful notes |
|---|---|---|---|---|---|---|
| Graph-check baseline | n/a | n/a | 36.1% | 0.328 | n/a | n/a |
