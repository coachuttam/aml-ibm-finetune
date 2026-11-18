# Base versus fine-tuned model (original alerts)

Typology accuracy and macro-F1 over the five labels, scored by the same functions as `python -m src.evaluate`, from the saved answers in `results/`.

| Split | System | Accuracy | Macro-F1 | Valid answers | Faithful reasons | Risk accuracy |
|---|---|---|---|---|---|---|
| validation | Base model | 0.153 | 0.119 | 87.3% | 96.3% | 26.7% |
| validation | Fine-tuned | 1.000 | 1.000 | 100.0% | 100.0% | 100.0% |
| test | Base model | 0.182 | 0.135 | 88.5% | 96.5% | 23.8% |
| test | Fine-tuned | 1.000 | 1.000 | 100.0% | 100.0% | 99.8% |
| held-out (saved answers, not re-run) | Base model | 0.187 | 0.181 | 72.0% | 93.0% | 23.3% |
| held-out (saved answers, not re-run) | Fine-tuned | 0.940 | 0.936 | 99.7% | 99.7% | 99.7% |
