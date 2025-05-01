#!/usr/bin/env python3
# End-to-end pipeline with new <|endq|> stop-token support
# --------------------------------------------------------------------------
import argparse, itertools, json, random, sys
import jsonlines, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def warn(msg): print(f"[WARN] {msg}", file=sys.stderr)
def after_anchor(txt, anchor): return txt.split(anchor, 1)[1].split("<|endq|>",1)[0].strip()

def load_lm(base, adapter=None):
    tok = AutoTokenizer.from_pretrained(base, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16,
                                               device_map="auto")
    mdl.resize_token_embeddings(len(tok)); mdl.eval()
    if adapter: mdl = PeftModel.from_pretrained(mdl, adapter, is_trainable=False)
    return mdl, tok

@torch.inference_mode()
def run(mdl, tok, prompt, cfg):
    ids = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**ids, **cfg)
    return tok.decode(out[0], skip_special_tokens=True)

def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            if not line.strip(): continue
            try: yield json.loads(line)
            except Exception as e: warn(f"bad JSON line {idx}: {e}")

# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question_file")
    ap.add_argument("--yaml_path", default="utils/load_abstract_db/output.yaml")
    ap.add_argument("--output_path", default="output.jsonl")
    ap.add_argument("--num_examples", type=int)
    ap.add_argument("--q_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--a_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--c_base", default="meta-llama/Llama-3.2-3B")
    ap.add_argument("--q_adapter_path", default="models/updated_db/ft_q_llama_r25")
    ap.add_argument("--a_adapter_path", default="models/updated_db/ft_a_llama_r24")
    args = ap.parse_args()

    # ---------------- models
    use_qfile = bool(args.question_file)
    if not use_qfile:
        q_m, q_t = load_lm(args.q_base, args.q_adapter_path)
        q_cfg = dict(max_new_tokens=50, temperature=0.7, top_k=50,
                     stopping_criteria=q_t("<|endq|>", add_special_tokens=False).input_ids)
    a_m, a_t = load_lm(args.a_base, args.a_adapter_path)
    c_m, c_t = load_lm(args.c_base)

    a_cfg = dict(max_new_tokens=100, temperature=0.6, top_k=40)
    c_cfg = dict(max_new_tokens=1, temperature=0.1, top_k=10)

    # ---------------- data stream
    if use_qfile:
        data_iter = iter_jsonl(args.question_file)
        total = sum(1 for _ in open(args.question_file, encoding="utf-8"))
    else:
        from utils.load_abstract_db.file_readers import read_output_yaml_file
        yaml = read_output_yaml_file(args.yaml_path); random.shuffle(yaml)
        data_iter = iter(yaml); total = len(yaml)

    if args.num_examples:
        data_iter = itertools.islice(data_iter, args.num_examples)
        total = min(total, args.num_examples)

    # ---------------- loop
    with jsonlines.open(args.output_path, "w") as out, \
         tqdm(total=total, desc="Processing") as bar:
        for rec in data_iter:
            doi = rec.get("doi") or ""
            if not isinstance(doi, str) or doi.lower() == "nan":
                warn("missing DOI → skip"); bar.update(); continue

            question = (rec.get("question") or "").strip()
            abstract = rec.get("abstract") or ""

            # ----- generate question (if absent)
            if not question and not use_qfile:
                q_prompt = ("Generate ONE research question.\n\n"
                            f"Abstract(s):\n{abstract}\n\nQuestion:")
                question = after_anchor(run(q_m, q_t, q_prompt, q_cfg), "Question:")

            if not question:
                warn("empty question → skip"); bar.update(); continue

            # ----- answer
            a_prompt = ("Answer based ONLY on the abstract(s). If insufficient info, say 'N/A'.\n\n"
                        f"Abstract(s):\n{abstract}\n\nQuestion: {question}\n\nAnswer:")
            answer = after_anchor(run(a_m, a_t, a_prompt, a_cfg), "Answer:")

            # one retry if N/A
            if "N/A" in answer:
                re_prompt = (f"{a_prompt}\n\nRevised answer:")
                answer = after_anchor(run(a_m, a_t, re_prompt, a_cfg), "Revised answer:")

            # ----- category
            c_prompt = ("Classify the answer as one of: method, knowledge, discussion.\n\n"
                        f"Question: {question}\nAnswer: {answer}\n\nCategory:")
            category = after_anchor(run(c_m, c_t, c_prompt, c_cfg), "Category:")

            out.write({"doi": doi, "question": question,
                       "answer": answer, "category": category})
            bar.update()

    print("✅  Results saved to", args.output_path)

if __name__ == "__main__":
    main()
