#!/usr/bin/env python
# Fine-tune Llama-3-3B with LoRA-8bit on the new dataset (question or answer)
# --------------------------------------------------------------------------
import argparse, os, random, math, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments,
                          DataCollatorForLanguageModeling,
                          EarlyStoppingCallback)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training, get_peft_model

# --- helpers ---------------------------------------------------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def perplexity(eval_pred):            # simple eval metric
    loss = eval_pred["loss"]
    return {"perplexity": math.exp(loss) if loss < 20 else float("inf")}

def load_jsonl(path):                 # HF datasets loader
    return load_dataset("json", data_files=path, split="train")

# --- main ------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["question", "answer"], required=True)
    p.add_argument("--data_dir",   default="data/finetune_data_v2")
    p.add_argument("--output_dir", default="ft_ckpt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); set_seed(args.seed)

    base = "meta-llama/Llama-3.2-3B"
    file = "question_dataset.jsonl" if args.task == "question" else "answer_dataset.jsonl"

    ds = load_jsonl(os.path.join(args.data_dir, file))
    val = max(1, int(0.2 * len(ds)))
    train_ds, val_ds = ds.train_test_split(test_size=val, seed=args.seed).values()

    # --- model & tokenizer --------------------------------------------------
    load_kw = dict(device_map="auto", load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    if "<|endq|>" not in tok.get_vocab():           # add special token once
        tok.add_special_tokens({"additional_special_tokens": ["<|endq|>"]})

    model = AutoModelForCausalLM.from_pretrained(base, **load_kw)
    model.resize_token_embeddings(len(tok))
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def tok_fn(ex):
        enc = tok(ex["prompt"] + ex["response"],
                  truncation=True, padding="max_length", max_length=768)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train_ds = train_ds.map(tok_fn, remove_columns=train_ds.column_names); train_ds.set_format("torch")
    val_ds   = val_ds  .map(tok_fn, remove_columns=val_ds.column_names);   val_ds  .set_format("torch")

    coll = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2,
        learning_rate=args.lr, warmup_ratio=0.1, lr_scheduler_type="cosine",
        gradient_accumulation_steps=4,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        weight_decay=0.05, logging_steps=50, fp16=False, report_to="none",
        seed=args.seed
    )

    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=coll, tokenizer=tok,
                      compute_metrics=perplexity,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=1)])

    trainer.train()
    trainer.save_model(args.output_dir); tok.save_pretrained(args.output_dir)
    print("✅ adapter saved →", args.output_dir)

if __name__ == "__main__":
    main()
