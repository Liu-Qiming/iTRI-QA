#!/usr/bin/env python3
# scripts/evaluation.py
# ------------------------------------------------------------
# Evaluate iTRI-QA outputs on three automatic metrics
#   • Question fluency  →  GPT-2 perplexity  (lower = better)
#   • Answer F1         →  SQuAD-v2 token F1 (higher = better)
#   • Category accuracy →  exact match       (higher = better)
# ------------------------------------------------------------
import argparse, jsonlines, math, torch, statistics, pathlib, sys
from collections import defaultdict

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_metric                # Hugging-Face SQuAD v2 metric


# --------------------------------------------------------------------- helpers
def load_jsonl(path):
    data = {}
    with jsonlines.open(path) as rdr:
        for rec in rdr:
            doi = str(rec.get("doi", "")).strip().lower()
            if doi:
                data[doi] = rec
    return data


def gpt2_perplexity(sentences, batch_size=16, device="cuda"):
    tok  = GPT2TokenizerFast.from_pretrained("gpt2-medium")
    mdl  = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
    mdl.eval()

    ppl_scores = []
    with torch.inference_mode():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc   = tok(batch, return_tensors="pt",
                        padding=True, truncation=True).to(device)
            out   = mdl(**enc, labels=enc["input_ids"])
            neg_ll = out.loss.detach().cpu().float().item()
            ppl_scores.append(math.exp(neg_ll))
    return statistics.mean(ppl_scores)


def squad_f1(predictions, references):
    """predictions/references: list[str] aligned one-to-one"""
    squad_metric = load_metric("squad_v2")
    formatted = [{"id": str(i), "prediction_text": p} for i,p in enumerate(predictions)]
    refs      = [{"id": str(i), "answers": {"text": [r], "answer_start":[0]}}
                 for i,r in enumerate(references)]
    res = squad_metric.compute(predictions=formatted, references=refs)
    return res["f1"]           # token-level F1


def accuracy(pred, ref):
    ok = sum(1 for p,r in zip(pred, ref) if p.strip().lower() == r.strip().lower())
    return ok / len(ref) if ref else 0.0


# --------------------------------------------------------------- main routine
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",  required=True, help="Reference JSONL (197 human QA)")
    ap.add_argument("--pred", required=True, help="Model output JSONL to score")
    args = ap.parse_args()

    ref  = load_jsonl(args.ref)
    pred = load_jsonl(args.pred)

    common = [doi for doi in ref if doi in pred]
    if not common:
        sys.exit("No DOI overlap between reference and prediction files!")

    # aligned lists ----------------------------------------------------------
    q_ref = [ref[d]["question"]          for d in common]
    q_pre = [pred[d]["question"]         for d in common]

    a_ref = [ref[d]["answer"]            for d in common]
    a_pre = [pred[d]["answer"]           for d in common]

    c_ref = [ref[d]["category"] or "none" for d in common]
    c_pre = [pred[d]["category"] or "none" for d in common]

    # metrics ----------------------------------------------------------------
    ppl  = gpt2_perplexity(q_pre)
    f1   = squad_f1(a_pre, a_ref)
    acc  = accuracy(c_pre, c_ref)

    # report -----------------------------------------------------------------
    print(f"# samples            : {len(common)}")
    print(f"Question perplexity  : {ppl:7.2f} ↓")
    print(f"Answer F1           : {f1:7.2f} ↑")
    print(f"Category accuracy   : {acc*100:7.2f}% ↑")


if __name__ == "__main__":
    main()
