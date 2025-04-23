#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline:
    •    generate OR read question            (LoRA optional)
    •    generate answer                      (LoRA optional)
    •    categorize                           (base model)

--question_file  lets you feed a JSONL produced by a previous run;
                 records must contain at least {"doi": "...", "question": "..."}.
"""

import argparse, os, random, jsonlines, torch
from pathlib import Path
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ───────────────────────── helpers ──────────────────────────
def after(text: str, anchor: str) -> str:
    return text.split(anchor, 1)[1].strip() if anchor in text else text.strip()


def load_lm(base: str, adapter: str | None = None):
    """
    Load *half-precision* base model; attach LoRA/PEFT adapter if supplied.
    """
    tok = AutoTokenizer.from_pretrained(base, use_fast=False)
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="auto"
    )
    model.resize_token_embeddings(len(tok))
    model.eval()

    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        name = Path(adapter).name
        assert name in model.peft_config, f"adapter {name} not found!"
        assert model.active_adapter == name
    return model, tok


@torch.inference_mode()
def generate(model, tok, prompt: str, cfg: dict) -> str:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ids, **cfg)
    return tok.decode(out[0], skip_special_tokens=True)


# ───────────────────────── main ─────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("multi-step QA pipeline")
    ap.add_argument("--yaml_path", default="utils/load_abstract_db/output.yaml")
    ap.add_argument("--output_path", default="output.jsonl")
    ap.add_argument("--num_examples", type=int)

    # optional pre-computed question file
    ap.add_argument("--question_file", help="JSONL with fields doi,question")

    # base models
    ap.add_argument("--q_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--a_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--c_base", default="meta-llama/Llama-3.2-3B")

    # adapters
    ap.add_argument("--q_adapter_path", default=None)
    ap.add_argument("--a_adapter_path", default=None)
    args = ap.parse_args()

    # 1)  load question map if provided
    q_map = {}
    if args.question_file and Path(args.question_file).is_file():
        with jsonlines.open(args.question_file) as rdr:
            for rec in rdr:
                if "doi" in rec and "question" in rec:
                    q_map[rec["doi"]] = rec["question"]

    need_q_model = not bool(q_map)
    if need_q_model:
        q_model, q_tok = load_lm(args.q_base, args.q_adapter_path)
    else:
        q_model = q_tok = None  # free memory

    # always need answer + category models
    a_model, a_tok = load_lm(args.a_base, args.a_adapter_path)
    c_model, c_tok = load_lm(args.c_base)

    # gen configs
    q_cfg = dict(max_new_tokens=50, temperature=0.7, top_k=50)
    a_cfg = dict(max_new_tokens=100, temperature=0.6, top_k=40)
    c_cfg = dict(max_new_tokens=1, temperature=0.1, top_k=10)

    # 2)  load abstracts
    from utils.load_abstract_db.file_readers import read_output_yaml_file

    data = read_output_yaml_file(args.yaml_path)
    random.shuffle(data)
    if args.num_examples:
        data = data[: args.num_examples]

    # 3)  pipeline loop
    with jsonlines.open(args.output_path, "w") as w:
        for item in tqdm(data, desc="Processing"):
            doi = item.get("doi", "unknown")
            abstract = item.get("abstract", "")

            # ── question ────────────────────────────────────────────────
            if doi in q_map:
                question = q_map[doi]
            else:
                if not need_q_model:  # q_model wasn't loaded but DOI missing
                    raise ValueError(f"Question for DOI {doi} not found in question_file.")
                anchor_q = "Question:"
                q_prompt = (
                    "You are a creative research assistant who generates thoughtful and meaningful "
                    "questions from one or more scientific abstracts. Only use the abstracts "
                    "provided below. Do not invent or assume information. Craft a specific, probing "
                    "question that arises strictly from these abstracts. Do NOT generate "
                    "multiple-choice, yes/no, true/false, or simple factual questions.\n\n"
                    f"Here are the abstracts:\n{abstract}\n\n{anchor_q}"
                )
                out = generate(q_model, q_tok, q_prompt, q_cfg)
                question = after(out, anchor_q)

            # ── answer ─────────────────────────────────────────────────
            anchor_a = "Answer:"
            a_prompt = (
                "You are an expert research assistant. Below are one or more scientific abstracts "
                "followed by a question. Provide a concise and insightful answer based only on the "
                "provided abstracts. If the abstracts do not contain enough information, respond "
                "'N/A'. Do not invent or assume facts beyond what the abstracts state.\n\n"
                f"Here are the abstracts:\n{abstract}\n\n"
                f"Question: {question}\n\n{anchor_a}"
            )
            ans = after(generate(a_model, a_tok, a_prompt, a_cfg), anchor_a)

            if "N/A" in ans:
                anchor_re = "Revised answer:"
                re_prompt = (
                    "It appears your answer may be incomplete or marked as 'N/A'. Please re-check "
                    "the abstracts and provide an answer if evidence exists; otherwise reply 'N/A'.\n\n"
                    f"Here are the abstracts:\n{abstract}\n\n"
                    f"Question: {question}\n\n{anchor_re}"
                )
                revised = after(generate(a_model, a_tok, re_prompt, a_cfg), anchor_re)
                if "N/A" not in revised:
                    ans = revised

            # ── categorize ─────────────────────────────────────────────
            anchor_c = "Category (choose only from method, knowledge, discussion):"
            c_prompt = (
                "You are a scientific reviewer. Given the following Q&A pair, classify the insight "
                "type as one of: method, knowledge, or discussion.\n\n"
                f"Question: {question}\nAnswer: {ans}\n\n{anchor_c}"
            )
            cat = after(generate(c_model, c_tok, c_prompt, c_cfg), anchor_c)

            w.write({"doi": doi, "question": question, "answer": ans, "category": cat})

    print("✅ pipeline finished →", args.output_path)


if __name__ == "__main__":
    main()
