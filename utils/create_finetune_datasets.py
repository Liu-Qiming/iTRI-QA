#!/usr/bin/env python3
# Re-generate fine-tuning JSONL files:
#   • question_dataset.jsonl   (adds <|endq|> terminator)
#   • answer_dataset.jsonl     (unchanged)
# ------------------------------------------------------------------
import argparse, os, jsonlines

ENDQ = "<|endq|>"

Q_PROMPT = (
    "Generate ONE research question that arises strictly from the abstract(s) "
    "below. Do NOT answer it.\n\n"
    "Abstract(s):\n{abs}\n\nQuestion:"
)
A_PROMPT = (
    "Answer the question using ONLY the information in the abstract(s).\n\n"
    "Abstract(s):\n{abs}\n\nQuestion: {q}\n\nAnswer:"
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",      default="data/new_database.jsonl")
    ap.add_argument("--output_dir", default="data/finetune_data_v2/")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    q_out = os.path.join(args.output_dir, "question_dataset.jsonl")
    a_out = os.path.join(args.output_dir, "answer_dataset.jsonl")

    with ( jsonlines.open(args.input) as rdr,
           jsonlines.open(q_out, "w") as qw,
           jsonlines.open(a_out, "w") as aw ):

        for rec in rdr:
            abs_block = "\n\n".join(rec["abstract"].values()).strip()
            q_txt     = rec["question"].strip()
            a_txt     = rec["answer"].strip()

            qw.write({
                "prompt":   Q_PROMPT.format(abs=abs_block),
                "response": q_txt + ENDQ          # explicit stop token
            })
            aw.write({
                "prompt":   A_PROMPT.format(abs=abs_block, q=q_txt),
                "response": a_txt
            })

    print("✅  Files written to", args.output_dir)

if __name__ == "__main__":
    main()
