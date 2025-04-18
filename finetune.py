#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA / QLoRA fine‑tuning script for smaller (≈3 B) causal‑LMs.

Usage examples
--------------
# Question model – LLaMA‑3.2‑3B – fp16
python finetune_peft_small.py \
  --task question \
  --model_name meta-llama/Llama-3.2-3B \
  --data_dir data/finetune_data \
  --output_dir ft_q_llama

# Answer model – Falcon‑3B – 8‑bit QLoRA
python finetune_peft_small.py \
  --task answer \
  --model_name tiiuae/Falcon-3B-Instruct \
  --bits 8 \
  --data_dir data/finetune_data \
  --output_dir ft_a_falcon
"""
import argparse, os, random, json, torch, numpy as np
from inspect import signature
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ----------------------------------------------------------------------------- #
# utilities
# ----------------------------------------------------------------------------- #
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str):
    return load_dataset("json", data_files=path, split="train")


def default_target_modules(model_name: str):
    name = model_name.lower()
    if any(k in name for k in ["llama", "alpaca", "mistral"]):
        return ["q_proj", "v_proj"]
    if "falcon" in name:
        return ["query_key_value"]
    # generic fallback
    return ["q_proj", "v_proj"]


# ----------------------------------------------------------------------------- #
# main training routine
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser("LoRA / QLoRA fine‑tuning (question|answer)")
    parser.add_argument("--task", choices=["question", "answer"], required=True)
    parser.add_argument("--model_name", required=True, help="HF hub ID or local path")
    parser.add_argument("--data_dir", default="data/finetune_data")
    parser.add_argument("--output_dir", default="ft_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=16, choices=[4, 8, 16])
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA overrides
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", nargs="*", default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    # dataset selection --------------------------------------------------------
    fname = "question_dataset.jsonl" if args.task == "question" else "answer_dataset.jsonl"
    ds_path = os.path.join(args.data_dir, fname)
    dataset = load_jsonl(ds_path)

    val_size = max(1, int(len(dataset) * args.val_split))
    train_ds, val_ds = dataset.train_test_split(test_size=val_size, seed=args.seed).values()

    # tokenizer / model --------------------------------------------------------
    load_kwargs = {"device_map": "auto"}
    if args.bits == 16:
        load_kwargs["torch_dtype"] = torch.float16
    else:  # 8‑bit / 4‑bit
        load_kwargs[f"load_in_{args.bits}bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
    if tokenizer.pad_token_id != model.config.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    if args.bits in (4, 8):
        model = prepare_model_for_kbit_training(model)

    # LoRA setup --------------------------------------------------------------
    tgt_modules = args.target_modules or default_target_modules(args.model_name)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=tgt_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # tokenisation ------------------------------------------------------------
    def tok_fn(ex):
        enc = tokenizer(
            ex["prompt"] + ex["response"],
            truncation=True,
            max_length=args.max_tokens,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()   # ← add labels for loss
        return enc


    train_ds = train_ds.map(tok_fn, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tok_fn, remove_columns=val_ds.column_names)
    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # TrainingArguments – version‑aware ---------------------------------------
    # ---------------------------------------------------------
    # build TrainingArguments kwargs version‑safely
    # ---------------------------------------------------------
    ta_sig = signature(TrainingArguments)

    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=(args.bits == 16),
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        gradient_accumulation_steps=4,
    )

    # evaluation handling
    if "evaluation_strategy" in ta_sig.parameters:
        # modern versions (>=4.3)
        kwargs.update(
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
        )
    elif "eval_steps" in ta_sig.parameters:
        # very old versions: fall back to eval_steps every epoch
        steps_per_epoch = max(1, len(train_ds) // args.batch_size)
        kwargs.update(eval_steps=steps_per_epoch)
        # load_best_model_at_end may not exist; add only if present
        if "load_best_model_at_end" in ta_sig.parameters:
            kwargs["load_best_model_at_end"] = False

    targs = TrainingArguments(**kwargs)

    # ---------------------------------------------------------
    # decide whether EarlyStopping is usable for this HF version
    # ---------------------------------------------------------
    use_early_stop = (
        "evaluation_strategy" in ta_sig.parameters       # new Trainer
        and kwargs.get("load_best_model_at_end", False)  # metric tracking enabled
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)] if use_early_stop else []

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        callbacks=callbacks,   # <‑‑ here
    )


    # train -------------------------------------------------------------------
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Finished fine‑tuning {args.task}. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
