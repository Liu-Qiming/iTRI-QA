#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline: (1) generate question  ➜  (2) generate answer  ➜  (3) categorise

• If --question_file is supplied and a record already has a non-empty "question",
  we skip step-1 (question generation).
• Any record that cannot be parsed, or is missing a DOI / question, is skipped.
• A tqdm progress-bar is always shown (total rows = line-count of the source
  file or len(yaml_data)).
"""

import argparse, json, random, sys, itertools, os
import jsonlines, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------------------------
def warn(msg: str):
    print(f"[WARN] {msg}", file=sys.stderr)

def extract_after_anchor(text: str, anchor: str) -> str:
    return text.split(anchor, 1)[1].strip() if anchor in text else text.strip()

def load_lm(base_name: str, adapter_path: str | None = None):
    tok = AutoTokenizer.from_pretrained(base_name, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.float16, device_map="auto"
    )
    mdl.resize_token_embeddings(len(tok))
    mdl.eval()
    if adapter_path:
        mdl = PeftModel.from_pretrained(mdl, adapter_path, is_trainable=False)
    return mdl, tok

@torch.inference_mode()
def generate(model, tok, prompt: str, cfg: dict) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, **cfg)
    return tok.decode(out[0], skip_special_tokens=True)

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_path",
                    default="utils/load_abstract_db/output.yaml",
                    help="YAML abstracts file (ignored if --question_file given)")
    ap.add_argument("--question_file",
                    help="Plain-text JSONL with existing questions; must provide "
                         "'doi' and 'question'. If omitted, questions are generated.")
    ap.add_argument("--output_path", default="output.jsonl")
    ap.add_argument("--num_examples", type=int, default=None)

    # model / adapter paths
    ap.add_argument("--q_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--a_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--c_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--q_adapter_path", default="models/updated_db/ft_q_llama_r24")
    ap.add_argument("--a_adapter_path", default="models/updated_db/ft_a_llama_r24")
    args = ap.parse_args()

    use_qfile = args.question_file is not None

    # ------------------------------  load models
    if not use_qfile:
        q_model, q_tok = load_lm(args.q_base, args.q_adapter_path)
        q_cfg = dict(max_new_tokens=50, temperature=0.7, top_k=50)
    else:
        q_model = q_tok = q_cfg = None

    a_model, a_tok = load_lm(args.a_base, args.a_adapter_path)
    c_model, c_tok = load_lm(args.c_base)

    a_cfg = dict(max_new_tokens=100, temperature=0.6, top_k=40)
    c_cfg = dict(max_new_tokens=1,   temperature=0.1, top_k=10)

    # ------------------------------  data stream
    if use_qfile:
        total_rows = sum(1 for _ in open(args.question_file, "r", encoding="utf-8"))

        def record_stream():
            with open(args.question_file, "r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        if not isinstance(rec, dict):
                            raise ValueError("not a JSON object")
                    except Exception as exc:          # <-- swallow errors
                        warn(f"Line {idx}: {exc} → skipped")
                        continue
                    yield rec
    else:
        from utils.load_abstract_db.file_readers import read_output_yaml_file
        yaml_data = read_output_yaml_file(args.yaml_path)
        random.shuffle(yaml_data)
        total_rows = len(yaml_data)

        def record_stream():
            for rec in yaml_data:
                yield rec

    # honour --num_examples
    stream = record_stream()
    if args.num_examples:
        stream = itertools.islice(stream, args.num_examples)
        total_rows = min(total_rows, args.num_examples)

    # ------------------------------  main loop
    with jsonlines.open(args.output_path, "w") as writer, \
         tqdm(total=total_rows, desc="Processing", unit="sample") as pbar:

        for rec in stream:
            raw_doi = rec.get("doi", "")
            question_txt = (rec.get("question") or "").strip()

            # ----- basic validity
            if not (isinstance(raw_doi, str) and raw_doi.strip()
                    and raw_doi.lower() != "nan"):
                warn("record without DOI – skipped")
                pbar.update(1); continue
            doi = raw_doi.strip()

            # ----- Question generation (if needed)
            if not question_txt:
                if use_qfile:           # question missing but we cannot generate
                    warn("question empty & no question model – skipped")
                    pbar.update(1); continue

                anchor_q = "Question:"
                q_prompt = (
                    "You are a creative research assistant who generates thoughtful and meaningful "
                    "questions from scientific abstracts. Only use the abstracts provided below. "
                    "Do not invent or assume information. Craft a specific, probing, and original "
                    "question that arises strictly from these abstracts. Do NOT generate multiple-"
                    "choice, yes/no, true/false, or simple factual questions.\n\n"
                    f"Here are the abstracts:\n{rec.get('abstract','')}\n\n{anchor_q}"
                )
                q_raw = generate(q_model, q_tok, q_prompt, q_cfg)
                question_txt = extract_after_anchor(q_raw, anchor_q)

            if not question_txt:
                warn("empty question after generation – skipped")
                pbar.update(1); continue

            # ----- Answer generation
            anchor_a = "Answer:"
            a_prompt = (
                "You are an expert research assistant. Below are one or more scientific abstracts "
                "followed by a question. Provide a concise and insightful answer based only on the "
                "provided abstracts. If the abstracts do not contain enough information, respond "
                "with 'N/A'. Do not invent or assume facts.\n\n"
                f"Here are the abstracts:\n{rec.get('abstract','')}\n\n"
                f"Question: {question_txt}\n\n{anchor_a}"
            )
            a_raw = generate(a_model, a_tok, a_prompt, a_cfg)
            answer_txt = extract_after_anchor(a_raw, anchor_a)

            # retry once if 'N/A'
            if "N/A" in answer_txt:
                anchor_re = "Revised answer:"
                re_prompt = (
                    "It appears your answer may be incomplete or marked as 'N/A'. "
                    "Please re-check the abstracts and answer if any evidence exists; "
                    "otherwise reply 'N/A'.\n\n"
                    f"Here are the abstracts:\n{rec.get('abstract','')}\n\n"
                    f"Question: {question_txt}\n\n{anchor_re}"
                )
                re_raw = generate(a_model, a_tok, re_prompt, a_cfg)
                re_txt = extract_after_anchor(re_raw, anchor_re)
                if "N/A" not in re_txt:
                    answer_txt = re_txt

            # ----- Categorisation
            anchor_c = "Category (choose only from method, knowledge, discussion):"
            c_prompt = (
                "You are a scientific reviewer. Given the following Q&A pair, classify the type "
                "of insight provided as one of: method, knowledge, or discussion. Choose the "
                "category that best fits the content of the answer.\n\n"
                f"Question: {question_txt}\nAnswer: {answer_txt}\n\n{anchor_c}"
            )
            c_raw = generate(c_model, c_tok, c_prompt, c_cfg)
            category = extract_after_anchor(c_raw, anchor_c)

            writer.write(
                {
                    "doi":      doi,
                    "question": question_txt,
                    "answer":   answer_txt,
                    "category": category,
                }
            )
            pbar.update(1)

    print(f"✅ Finished – results saved to {args.output_path}")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
