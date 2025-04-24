#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline:  (1) generate question  ➜  (2) generate answer  ➜  (3) categorise
If --question_file is provided *and* it already contains a `question` field,
we skip step-1 entirely and go straight to answer generation.

Rows with bad JSON, missing DOI, or (when needed) missing question
are silently skipped (a warning is printed to stderr).
"""

import argparse, random, sys, json, jsonlines, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr)

def extract_after_anchor(text: str, anchor: str) -> str:
    return text.split(anchor, 1)[1].strip() if anchor in text else text.strip()

def load_lm(base_name: str, adapter_path: str | None = None):
    """
    Load a Causal-LM + tokenizer (half-precision), optionally attach LoRA adapter.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

    return model, tokenizer

@torch.inference_mode()
def generate(model, tok, prompt: str, gen_cfg: dict) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **gen_cfg)
    return tok.decode(out[0], skip_special_tokens=True)

# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser("QA pipeline w/ optional LoRA adapters")
    ap.add_argument("--yaml_path",
                    default="utils/load_abstract_db/output.yaml",
                    help="YAML abstracts file (ignored if --question_file given)")
    ap.add_argument("--question_file",
                    help="JSONL with existing questions; must contain 'doi' and "
                         "'question'. If omitted, questions are generated.")
    ap.add_argument("--output_path", default="output.jsonl")
    ap.add_argument("--num_examples", type=int, default=None)

    # base checkpoints
    ap.add_argument("--q_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--a_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--c_base", default="meta-llama/Llama-3.2-3B")

    # adapter dirs (optional)
    ap.add_argument("--q_adapter_path",
                    default="models/updated_db/ft_q_llama_r24",
                    help="LoRA adapter for question model (or omit)")
    ap.add_argument("--a_adapter_path",
                    default="models/updated_db/ft_a_llama_r24",
                    help="LoRA adapter for answer model (or omit)")

    args = ap.parse_args()

    # Decide whether we need the question model -----------------------------
    use_question_file = args.question_file is not None
    if not use_question_file:
        # will need to generate questions
        q_model, q_tok = load_lm(args.q_base, args.q_adapter_path)
        q_cfg = dict(max_new_tokens=50, temperature=0.7, top_k=50)
    else:
        q_model = q_tok = q_cfg = None  # unused

    # Always need answer + category models ----------------------------------
    a_model, a_tok = load_lm(args.a_base, args.a_adapter_path)
    c_model, c_tok = load_lm(args.c_base)

    a_cfg = dict(max_new_tokens=100, temperature=0.6, top_k=40)
    c_cfg = dict(max_new_tokens=1,   temperature=0.1, top_k=10)

    # ---------------------------------------------------------------------- #
    # Load records ---------------------------------------------------------- #
    if use_question_file:
        def record_stream():
            with jsonlines.open(args.question_file, "r") as rdr:
                for idx, rec in enumerate(rdr, 1):
                    if not isinstance(rec, dict):
                        warn(f"Line {idx}: not a JSON object – skipped")
                        continue
                    yield rec
    else:
        # lazy import of YAML reader
        from utils.load_abstract_db.file_readers import read_output_yaml_file
        raw = read_output_yaml_file(args.yaml_path)
        random.shuffle(raw)
        def record_stream():
            for rec in raw:
                yield rec

    # optional truncation
    stream = record_stream()
    if args.num_examples:
        from itertools import islice
        stream = islice(stream, args.num_examples)

    # ---------------------------------------------------------------------- #
    with jsonlines.open(args.output_path, "w") as writer:
        for rec in tqdm(stream, desc="Processing"):
            raw_doi = rec.get("doi", "")
            question = rec.get("question", "").strip()

            # basic validation: DOI must exist
            if not (isinstance(raw_doi, str) and raw_doi.strip() and
                    raw_doi.lower() != "nan"):
                warn("missing/invalid DOI → skipped")
                continue
            doi = raw_doi.strip()

            # ------------------------------------------------------------------
            # Step-1  question (if needed)
            # ------------------------------------------------------------------
            if not use_question_file:          # need to generate question
                abstract = rec.get("abstract", "")
                anchor_q = "Question:"
                q_prompt = (
                    "You are a creative research assistant who generates thoughtful and "
                    "meaningful questions from one or more scientific abstracts. "
                    "Only use the abstracts provided below. Do not invent or assume "
                    "information. Craft a specific, probing, and original question that "
                    "arises strictly from these abstracts. Do NOT generate multiple-choice, "
                    "yes/no, true/false, or simple factual questions.\n\n"
                    f"Here are the abstracts:\n{abstract}\n\n{anchor_q}"
                )
                q_raw = generate(q_model, q_tok, q_prompt, q_cfg)
                question = extract_after_anchor(q_raw, anchor_q)

            # If still no question (e.g., empty field in file) → skip
            if not question:
                warn("missing question text → skipped")
                continue

            # ------------------------------------------------------------------
            # Step-2  answer
            # ------------------------------------------------------------------
            abstract = rec.get("abstract", "")
            anchor_a = "Answer:"
            a_prompt = (
                "You are an expert research assistant. Below are one or more scientific "
                "abstracts followed by a question. Provide a concise and insightful answer "
                "based only on the provided abstracts. If the abstracts do not contain "
                "enough information, respond with 'N/A'. Do not invent or assume facts.\n\n"
                f"Here are the abstracts:\n{abstract}\n\n"
                f"Question: {question}\n\n{anchor_a}"
            )
            a_raw = generate(a_model, a_tok, a_prompt, a_cfg)
            answer = extract_after_anchor(a_raw, anchor_a)

            # quick retry if N/A
            if "N/A" in answer:
                anchor_re = "Revised answer:"
                re_prompt = (
                    "It appears your answer may be incomplete or marked as 'N/A'. "
                    "Please re-check the abstracts and answer if any evidence exists; "
                    "otherwise reply 'N/A'.\n\n"
                    f"Here are the abstracts:\n{abstract}\n\n"
                    f"Question: {question}\n\n{anchor_re}"
                )
                re_raw = generate(a_model, a_tok, re_prompt, a_cfg)
                re_text = extract_after_anchor(re_raw, anchor_re)
                if "N/A" not in re_text:
                    answer = re_text

            # ------------------------------------------------------------------
            # Step-3  categorise
            # ------------------------------------------------------------------
            anchor_c = "Category (choose only from method, knowledge, discussion):"
            c_prompt = (
                "You are a scientific reviewer. Given the following Q&A pair, classify "
                "the type of insight provided as one of: method, knowledge, or discussion. "
                "Choose the category that best fits the content of the answer.\n\n"
                f"Question: {question}\nAnswer: {answer}\n\n{anchor_c}"
            )
            c_raw = generate(c_model, c_tok, c_prompt, c_cfg)
            category = extract_after_anchor(c_raw, anchor_c)

            # ------------------------------------------------------------------
            writer.write(
                {
                    "doi":      doi,
                    "question": question,
                    "answer":   answer,
                    "category": category,
                }
            )

    print(f"✅ Pipeline finished. Results in {args.output_path}")


if __name__ == "__main__":
    main()
