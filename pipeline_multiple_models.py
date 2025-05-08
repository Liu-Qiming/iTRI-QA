#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline:   (1) question  →  (2) answer  →  (3) category

▪ When --question_file is given and record["question"] is non-empty
  we skip (1).
▪ Records without a DOI are kept: we assign an in-memory UID.
▪ tqdm progress-bar always reflects the total lines processed.
"""

import argparse, itertools, json, os, random, sys, uuid
from pathlib import Path

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------------------------#
# Helpers
# ---------------------------------------------------------------------------#
def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def safe_str(v) -> str:
    """Return `v` if it is a non-NaN string; else an empty string."""
    if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
        return v.strip()
    return ""


def extract_after_anchor(text: str, anchor: str) -> str:
    """Keep everything after `anchor`, stop at 'Answer:' or blank line."""
    if anchor not in text:
        return text.strip()
    tail = text.split(anchor, 1)[1]
    for stop in ("Answer:", "\n\n", "\nAnswer"):
        if stop in tail:
            tail = tail.split(stop, 1)[0]
    return tail.strip()


def load_lm(backbone: str, adapter: str | None = None):
    tok = AutoTokenizer.from_pretrained(backbone, use_fast=False)
    tok.pad_token = tok.pad_token or tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        backbone, device_map="auto", torch_dtype=torch.float16
    )
    mdl.resize_token_embeddings(len(tok))
    mdl.eval()

    if adapter:
        mdl = PeftModel.from_pretrained(mdl, adapter, is_trainable=False)
    return mdl, tok


@torch.inference_mode()
def generate(m, t, prompt: str, gen_cfg: dict) -> str:
    ids = t(prompt, return_tensors="pt").to(m.device)
    out = m.generate(**ids, **gen_cfg)
    return t.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------#
# Main
# ---------------------------------------------------------------------------#
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_path", default="utils/load_abstract_db/output.yaml")
    ap.add_argument("--question_file")
    ap.add_argument("--output_path", default="output.jsonl")
    ap.add_argument("--num_examples", type=int)

    # backbone + adapters ----------------------------------------------------
    ap.add_argument("--q_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--a_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--c_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--q_adapter_path", default="models/updated_db/ft_q_llama_r24")
    ap.add_argument("--a_adapter_path", default="models/updated_db/ft_a_llama_r24")
    args = ap.parse_args()

    use_qfile = args.question_file is not None

    # --------------------  load models -------------------------------------
    if use_qfile:
        q_model = q_tok = q_cfg = None
    else:
        q_model, q_tok = load_lm(args.q_base, args.q_adapter_path)
        q_cfg = dict(max_new_tokens=50, temperature=0.7, top_k=50)

    a_model, a_tok = load_lm(args.a_base, args.a_adapter_path)
    c_model, c_tok = load_lm(args.c_base)

    a_cfg = dict(max_new_tokens=100, temperature=0.6, top_k=40)
    c_cfg = dict(max_new_tokens=1,   temperature=0.1, top_k=10)

    # --------------------  data stream -------------------------------------
    if use_qfile:
        total = sum(1 for _ in Path(args.question_file).open())
        def record_stream():
            with open(args.question_file, encoding="utf-8") as fh:
                for idx, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict):
                            raise ValueError("not a JSON object")
                        yield rec
                    except Exception as exc:
                        warn(f"Line {idx}: {exc} – skipped")
    else:
        from utils.load_abstract_db.file_readers import read_output_yaml_file
        yaml_data = read_output_yaml_file(args.yaml_path)
        random.shuffle(yaml_data)
        total = len(yaml_data)
        record_stream = lambda: iter(yaml_data)   # noqa: E731

    stream = record_stream()
    if args.num_examples:
        stream = itertools.islice(stream, args.num_examples)
        total = min(total, args.num_examples)

    # --------------------  main loop ---------------------------------------
    uid_counter = itertools.count(1)

    with jsonlines.open(args.output_path, "w") as writer, \
         tqdm(total=total, desc="Processing", unit="sample") as bar:

        for rec in stream:
            # -------- DOI handling -----------------------------------------
            doi = safe_str(rec.get("doi"))
            if not doi:
                doi = f"uid-{next(uid_counter)}" if use_qfile else str(uuid.uuid4())
                warn(f"missing DOI – assigned {doi}")

            # -------- question ---------------------------------------------
            question = safe_str(rec.get("question"))
            if not question and q_model is None:
                warn("question empty & Q-adapter disabled – skipped")
                bar.update(1); continue

            if not question:                                  # generate new Q
                anchor_q = "Question:"
                q_prompt = (
                    "You are a creative research assistant who generates a thoughtful and meaningful "
                    "question from scientific abstracts. Only use the abstracts provided below. "
                    "Do not invent or assume information. Craft a specific, probing, and original "
                    "question that arises strictly from these abstracts. Do NOT generate multiple-"
                    "choice, yes/no, true/false, or simple factual questions.\n\n"
                    f"Here are the abstracts:\n{rec.get('abstract','')}\n\n{anchor_q}"
                )
                raw_q = generate(q_model, q_tok, q_prompt, q_cfg)
                question = extract_after_anchor(raw_q, anchor_q)

            if not question:
                warn("empty question after generation – skipped")
                bar.update(1); continue

            # -------- answer -----------------------------------------------
            anchor_a = "Answer:"
            a_prompt = (
                "You are an expert research assistant. Below are one or more scientific abstracts "
                "followed by a question. Provide a concise and insightful answer based only on the "
                "provided abstracts. If the abstracts do not contain enough information, respond "
                "with 'N/A'. Do not invent or assume facts.\n\n"
                f"Here are the abstracts:\n{rec.get('abstract','')}\n\n"
                f"Question: {question}\n\n{anchor_a}"
            )
            raw_a = generate(a_model, a_tok, a_prompt, a_cfg)
            answer = extract_after_anchor(raw_a, anchor_a)

            # single retry if 'N/A'
            if "N/A" in answer:
                anchor_re = "Revised answer:"
                re_prompt = (
                    "It appears your answer may be incomplete or marked as 'N/A'. "
                    "Please re-check the abstracts and answer if any evidence exists; "
                    "otherwise reply 'N/A'.\n\n"
                    f"Here are the abstracts:\n{rec.get('abstract','')}\n\n"
                    f"Question: {question}\n\n{anchor_re}"
                )
                raw_re = generate(a_model, a_tok, re_prompt, a_cfg)
                re_ans = extract_after_anchor(raw_re, anchor_re)
                if "N/A" not in re_ans:
                    answer = re_ans

            # -------- category ---------------------------------------------
            anchor_c = "Category (choose only from method, knowledge, discussion):"
            c_prompt = (
                "You are a scientific reviewer. Given the following Q&A pair, classify the type "
                "of insight provided as one of: method, knowledge, or discussion. Choose the "
                "category that best fits the content of the answer.\n\n"
                f"Question: {question}\nAnswer: {answer}\n\n{anchor_c}"
            )
            raw_c = generate(c_model, c_tok, c_prompt, c_cfg)
            category = extract_after_anchor(raw_c, anchor_c)

            # -------- write -------------------------------------------------
            writer.write({
                "doi": doi,
                "question": question,
                "answer": answer,
                "category": category,
            })
            bar.update(1)

    print(f"✅ Finished – saved to {args.output_path}")


# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
