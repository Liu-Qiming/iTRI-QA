#!/usr/bin/env python3
# Evaluate QA outputs with ROUGE-L, BERTScore (+ QuestEval if installed)

import argparse, json, sys
from pathlib import Path
from collections import defaultdict

from evaluate import load             # built-in metrics

# ---------------------------------------------------------------------
def read_jsonl(path):
    """Return {question : answer}."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            out[rec["question"].strip()] = rec["answer"].strip()
    return out

# ---------------------------------------------------------------------
def rouge_metrics(preds, refs):
    metric = load("rouge")
    p, r = [], []
    for q, gold in refs.items():
        p.append(preds.get(q, ""))
        r.append(gold)
    return metric.compute(
        predictions=p,
        references=r,
        rouge_types=["rougeL"]
    )["rougeL"] * 100       # %

def bertscore_metrics(preds, refs):
    metric = load("bertscore")
    p, r = [], []
    for q, gold in refs.items():
        p.append(preds.get(q, ""))
        r.append(gold)
    scores = metric.compute(
        predictions=p,
        references=r,
        lang="en",
        model_type="roberta-large"
    )
    return sum(scores["f1"]) / len(scores["f1"]) * 100

def questeval_metrics(preds, refs):
    try:
        metric = load("questeval")
    except FileNotFoundError:
        return None          # QuestEval not installed
    pred_list, ref_list = [], []
    for q, gold in refs.items():
        pred_list.append(preds.get(q, ""))
        ref_list.append(gold)
    return metric.compute(
        predictions=pred_list,
        list_references=ref_list
    )["score"] * 100

# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",  required=True, help="gold answers jsonl")
    ap.add_argument("--pred", required=True, help="system answers jsonl")
    args = ap.parse_args()

    refs  = read_jsonl(args.ref)
    preds = read_jsonl(args.pred)

    rougeL = rouge_metrics(preds, refs)
    bertF  = bertscore_metrics(preds, refs)
    quest  = questeval_metrics(preds, refs)

    print(f"ROUGE-L   : {rougeL:5.2f}")
    print(f"BERTScore : {bertF:5.2f}")
    if quest is not None:
        print(f"QuestEval : {quest:5.2f}")
    else:
        print("QuestEval : (metric not installed)")
    print(f"N         : {len(refs)}")

if __name__ == "__main__":
    main()
