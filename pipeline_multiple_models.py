#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline: generate question ➜ generate answer ➜ categorize
Supports loading PEFT/LoRA adapters for question & answer models.
"""

import argparse, random, jsonlines, torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def extract_after_anchor(text: str, anchor: str) -> str:
    return text.split(anchor, 1)[1].strip() if anchor in text else text.strip()


def load_lm(base_name: str, adapter_path: str | None = None):
    """
    Load base model + tokenizer in half‑precision; attach LoRA adapter if given.
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
    parser = argparse.ArgumentParser("QA pipeline with optional LoRA adapters")
    parser.add_argument("--yaml_path", default="utils/load_abstract_db/output.yaml")
    parser.add_argument("--output_path", default="output.jsonl")
    parser.add_argument("--num_examples", type=int, default=None)

    # base checkpoints
    parser.add_argument("--q_base", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--a_base", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--c_base", default="meta-llama/Llama-3.2-3B")

    # adapter dirs
    parser.add_argument(
        "--q_adapter_path",
        default="models/updated_db/ft_q_llama_3B_3epoch/ft_q_llama",
        help="LoRA adapter for question model (or omit)",
    )
    parser.add_argument(
        "--a_adapter_path",
        default="models/updated_db/ft_a_llama_3B_3epoch/ft_a_llama",
        help="LoRA adapter for answer model (or omit)",
    )
    args = parser.parse_args()

    # lazy import of YAML reader (your original helper)
    from utils.load_abstract_db.file_readers import read_output_yaml_file

    # load models ------------------------------------------------------------
    q_model, q_tok = load_lm(args.q_base, args.q_adapter_path)
    a_model, a_tok = load_lm(args.a_base, args.a_adapter_path)
    c_model, c_tok = load_lm(args.c_base)  # categorization (no adapter yet)

    # gen configs
    q_cfg = dict(max_new_tokens=50, temperature=0.7, top_k=50)
    a_cfg = dict(max_new_tokens=100, temperature=0.6, top_k=40)
    c_cfg = dict(max_new_tokens=1, temperature=0.1, top_k=10)

    # ---------------------------------------------------------------------- #
    yaml_data = read_output_yaml_file(args.yaml_path)
    random.shuffle(yaml_data)
    if args.num_examples:
        yaml_data = yaml_data[: args.num_examples]

    with jsonlines.open(args.output_path, "w") as writer:
        for item in tqdm(yaml_data, desc="Processing"):
            doi = item.get("doi", "unknown")
            abstract = item.get("abstract", "")

            # ---------------- question ------------------------------------
            anchor_q = "Question:"
            q_prompt = (
                "You are a creative research assistant who generates thoughtful and meaningful "
                "questions from one or more scientific abstracts. Only use the abstracts provided "
                "below. Do not invent or assume information. Craft a specific, probing, and "
                "original question that arises strictly from these abstracts. Do NOT generate "
                "multiple-choice, yes/no, true/false, or simple factual questions.\n\n"
                f"Here are the abstracts:\n{abstract}\n\n{anchor_q}"
            )
            q_raw = generate(q_model, q_tok, q_prompt, q_cfg)
            question_text = extract_after_anchor(q_raw, anchor_q)

            # ---------------- answer --------------------------------------
            anchor_a = "Answer:"
            a_prompt = (
                "You are an expert research assistant. Below are one or more scientific abstracts "
                "followed by a question. Provide a concise and insightful answer based only on the "
                "provided abstracts. If the abstracts do not contain enough information, respond "
                "with 'N/A'. Do not invent or assume facts beyond what the abstracts state.\n\n"
                f"Here are the abstracts:\n{abstract}\n\n"
                f"Question: {question_text}\n\n{anchor_a}"
            )
            a_raw = generate(a_model, a_tok, a_prompt, a_cfg)
            answer_text = extract_after_anchor(a_raw, anchor_a)

            # one re‑evaluation if answer == N/A
            if "N/A" in answer_text:
                anchor_re = "Revised answer:"
                re_prompt = (
                    "It appears your answer may be incomplete or marked as 'N/A'. Please re‑check "
                    "the abstracts and provide an answer if any evidence exists; otherwise reply "
                    "'N/A'.\n\n"
                    f"Here are the abstracts:\n{abstract}\n\n"
                    f"Question: {question_text}\n\n{anchor_re}"
                )
                re_raw = generate(a_model, a_tok, re_prompt, a_cfg)
                re_text = extract_after_anchor(re_raw, anchor_re)
                if "N/A" not in re_text:
                    answer_text = re_text

            # ---------------- categorize ----------------------------------
            anchor_c = "Category (choose only from method, knowledge, discussion):"
            c_prompt = (
                "You are a scientific reviewer. Given the following Q&A pair, classify the type "
                "of insight provided as one of: method, knowledge, or discussion. Choose the "
                "category that best fits the content of the answer.\n\n"
                f"Question: {question_text}\nAnswer: {answer_text}\n\n{anchor_c}"
            )
            c_raw = generate(c_model, c_tok, c_prompt, c_cfg)
            category = extract_after_anchor(c_raw, anchor_c)

            writer.write(
                {
                    "doi": doi,
                    "question": question_text,
                    "answer": answer_text,
                    "category": category,
                }
            )

    print(f"✅ Pipeline finished. Results in {args.output_path}")


if __name__ == "__main__":
    main()
