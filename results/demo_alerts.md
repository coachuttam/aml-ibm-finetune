# Demo alerts

Pre-run investigations are cached in `results/demo_cache/`; nothing is filed or sent.

- **clear fan-out**: `IBM-5078023`. rank 114 of the model's test alerts, score 1.000; the graph check (The sender paid 9 different accounts in 7 days.) and the patterns file agree.
- **cycle**: `IBM-5077466`. rank 73 of the model's test alerts, score 1.000; the graph check (Money paid to the receiver reaches the sender again after 3 payments.) and the patterns file agree.
- **borderline**: `IBM-4355742`. rank 10 of the model's test alerts, score 1.000; the model flags it but the patterns file says it is not laundering, so the suggested typology (GATHER_SCATTER) is a mismatch.
