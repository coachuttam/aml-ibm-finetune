"""QLoRA fine-tuning on the IBM typology examples. Runs on a GPU (Google Colab T4), not on the laptop.

    python -m src.ibm.finetune_ibm --data-folder data/ibm_sft --output-dir /content/drive/MyDrive/aml-ibm-finetune/run1

Same base model and QLoRA settings as src/finetune.py (it is imported, not copied). Differences:
the prompt and answer come from src/ibm/sft_format.py, inputs are longer so the sequence limit is
higher, and batches are smaller to fit a T4. Only train.jsonl is learned from; val.jsonl only
watches the loss. test.jsonl is never read here.
If Colab disconnects, run the same command again: it resumes from the newest checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from src.finetune import (
    BASE_MODEL,
    IGNORE_INDEX,
    LEARNING_RATE,
    last_checkpoint,
    load_model_and_tokenizer,
    pad_batch,
)
from src.ibm.build_sft import OUTPUT_FOLDER, load_examples
from src.ibm.sft_format import build_messages

MAX_SEQ_LENGTH = 1280  # inputs plus system prompt plus answer measure about 800 tokens (digits cost one token each)
EPOCHS = 3  # small dataset, so a few passes; watch eval_loss and stop if it rises
BATCH_SIZE = 2  # long inputs: batch 2 fits a 15 GB T4
EVAL_BATCH_SIZE = 2
ACCUMULATION_STEPS = 8  # effective batch of 16, the same as the rupee run
SAVE_EVERY_STEPS = 25  # about every 4 minutes; a disconnect loses little
WARMUP_STEPS = 10  # a short ramp-up; about 4% of the run
VALIDATION_LIMIT = 150  # evaluating on every validation row each time would cost minutes


def tokenize_example(example: dict, tokenizer, max_length: int = MAX_SEQ_LENGTH) -> Dict[str, List[int]]:
    """Token ids for one example, with the loss switched on for the answer only.

    Raises if the example is too long: cutting it would cut off the answer the model must learn.
    """
    prompt_text = tokenizer.apply_chat_template(
        build_messages(example["input"]), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(example["answer"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_length:
        raise ValueError(
            "%s is %d tokens; raise MAX_SEQ_LENGTH" % (example["alert_id"], len(prompt_ids) + len(answer_ids))
        )
    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def train(data_folder: Path, output_dir: Path, base_model: str, epochs: int, limit: Optional[int]) -> None:
    """Fine-tune and save the adapter into output_dir/adapter."""
    from transformers import Trainer, TrainingArguments

    model, tokenizer = load_model_and_tokenizer(base_model)
    train_rows = load_examples("train", data_folder)[:limit]
    validation_rows = load_examples("val", data_folder)[: (limit or VALIDATION_LIMIT)]
    train_data = [tokenize_example(row, tokenizer) for row in train_rows]
    validation_data = [tokenize_example(row, tokenizer) for row in validation_rows]
    print("Longest example: %d tokens" % max(len(item["input_ids"]) for item in train_data))
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        fp16=True,
        logging_steps=5,
        save_steps=SAVE_EVERY_STEPS,
        eval_steps=SAVE_EVERY_STEPS,
        eval_strategy="steps",
        save_total_limit=2,
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=lambda batch: pad_batch(batch, tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=last_checkpoint(output_dir))
    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    print("Adapter saved to %s" % (output_dir / "adapter"))


def main() -> None:
    """Read the options and start training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-folder", default=str(OUTPUT_FOLDER))
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=None, help="use only N examples (quick smoke test)")
    args = parser.parse_args()
    train(Path(args.data_folder), Path(args.output_dir), args.base_model, args.epochs, args.limit)


if __name__ == "__main__":
    main()
