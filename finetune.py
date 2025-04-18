import argparse, os, random, json, torch, numpy as np
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# -----------------------------------------------------------------------------
# utility helpers
# -----------------------------------------------------------------------------

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load_jsonl(path: str):
    return load_dataset("json", data_files=path, split="train")


def default_target_modules(model_name: str):
    name = model_name.lower()
    if any(k in name for k in ["llama", "mistral", "alpaca"]):
        return ["q_proj", "v_proj"]
    if "falcon" in name:
        return ["query_key_value"]
    # fallback to common linear proj names; user can override
    return ["q_proj", "v_proj"]

# -----------------------------------------------------------------------------
# main fine‑tune routine
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("LoRA / QLoRA fine‑tuning for 3‑B models (question/answer task)")
    parser.add_argument("--task", choices=["question", "answer"], required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_dir", default="data/finetune_data")
    parser.add_argument("--output_dir", default="ft_ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=16, choices=[4, 8, 16])
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    # LoRA hyper‑params overrides
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--target_modules", nargs="*", default=None,
                        help="Explicit list of modules for LoRA (optional)")
    args = parser.parse_args()

    set_seed(args.seed)

    # select dataset file
    file_map = {"question": "question_dataset.jsonl", "answer": "answer_dataset.jsonl"}
    ds_path = os.path.join(args.data_dir, file_map[args.task])
    dataset = load_jsonl(ds_path)

    val_sz = max(1, int(len(dataset) * args.val_split))
    train_ds, val_ds = dataset.train_test_split(test_size=val_sz, seed=args.seed).values()

    # model/tokenizer load
    load_kwargs = {"device_map": "auto"}
    if args.bits == 16:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        load_kwargs[f"load_in_{args.bits}bit"] = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    # ------------------------------------------------------------------
    # ensure a pad token exists
    # ------------------------------------------------------------------
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # or tokenizer.unk_token
        # make sure model knows the new voc size
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **load_kwargs)


    if args.bits in (4, 8):
        model = prepare_model_for_kbit_training(model)

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

    def tok_fn(ex):
        return tokenizer(ex["prompt"] + ex["response"], truncation=True, max_length=args.max_tokens, padding="max_length")

    train_ds = train_ds.map(tok_fn, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tok_fn, remove_columns=val_ds.column_names)
    train_ds.set_format("torch"); val_ds.set_format("torch")

    kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=(args.bits == 16),
        logging_steps=25,
        save_strategy="epoch",
        # old versions need this instead of evaluation_strategy
        # eval will still run each epoch because we pass eval_dataset to Trainer
        save_total_limit=1,
        seed=args.seed,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        gradient_accumulation_steps=4,
    )

    # only add evaluation_strategy if present in this transformers version
    from inspect import signature
    if "evaluation_strategy" in signature(TrainingArguments).parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["evaluate_during_training"] = True  # fallback for v<4.3

    targs = TrainingArguments(**kwargs)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Finished fine‑tuning {args.task} model. Saved to {args.output_dir}")

if __name__ == "__main__":
    main()