#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA / QLoRA fine‑tuning for **meta‑llama/Llama‑3.2‑3B** on ≈5 k domain samples.
Designed for a single 40 GB A100 GPU.

Key defaults
-------------
* **QLoRA‑8 bit** loading (fits comfortably).
* **LoRA r = 32, α = 64, dropout = 0.1** → enough capacity for small corpora.
* **1 epoch** + EarlyStopping(patience = 1) + **val‑split 20 %**.
* Cosine LR 3 e‑4, warm‑up 10 %, weight‑decay 0.05.
* Effective batch = 32 (tokens ≤ 768) via `batch_size 8 × grad_acc 4`.

Use two prepared datasets:
`question_dataset.jsonl` or `answer_dataset.jsonl` under `data/finetune_data/`,
each line: `{ "prompt": "…", "response": "…" }`.

Example (question fine‑tune)
---------------------------
```bash
python finetune_llama3_lora.py \
  --task question \
  --data_dir data/finetune_data \
  --output_dir models/ft_q_llama3
```
"""
import argparse, os, random, math, torch, numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    TaskType,
)

# ------------------------------------------------------------------
# util
# ------------------------------------------------------------------

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_jsonl(path: str):
    return load_dataset("json", data_files=path, split="train")


# ------------------------------------------------------------------
# metric
# ------------------------------------------------------------------

def perplexity(eval_pred):
    loss = eval_pred["loss"]
    try:
        ppl = math.exp(loss)
    except OverflowError:
        ppl = float("inf")
    return {"perplexity": ppl}


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["question", "answer"], required=True)
    p.add_argument("--data_dir", default="data/finetune_data")
    p.add_argument("--output_dir", default="ft_ckpt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--max_tokens", type=int, default=768)
    p.add_argument("--bits", type=int, default=8, choices=[4, 8, 16])
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    # optional overrides
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    args = p.parse_args()

    set_seed(args.seed)

    base_model = "meta-llama/Llama-3.2-3B"

    # ---------------- data ---------------------
    fname = "question_dataset.jsonl" if args.task == "question" else "answer_dataset.jsonl"
    ds_path = os.path.join(args.data_dir, fname)
    dataset = load_jsonl(ds_path)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_ds, val_ds = dataset.train_test_split(test_size=val_size, seed=args.seed).values()

    # ---------------- model/tokenizer ----------
    load_kw = {"device_map": "auto"}
    if args.bits == 16:
        load_kw["torch_dtype"] = torch.float16
    else:
        load_kw[f"load_in_{args.bits}bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kw)
    model.resize_token_embeddings(len(tokenizer))

    if args.bits in (4, 8):
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],  # llama3
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---------------- tokenisation ------------
    def tok(ex):
        enc = tokenizer(
            ex["prompt"] + ex["response"],
            max_length=args.max_tokens,
            truncation=True,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_ds = train_ds.map(tok, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tok, remove_columns=val_ds.column_names)
    train_ds.set_format("torch"); val_ds.set_format("torch")

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ---------------- training args -----------
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_perplexity",
        seed=args.seed,
        fp16=(args.bits == 16),
        gradient_accumulation_steps=4,
        save_total_limit=1,
        logging_steps=25,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=perplexity,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Saved adapter to {args.output_dir}")

if __name__ == "__main__":
    main()