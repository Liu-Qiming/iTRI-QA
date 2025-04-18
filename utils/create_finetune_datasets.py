import jsonlines
import argparse
import os

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/new_database.jsonl",           # ← your combined file in JSONL format
        help="Path to the combined JSONL file."
    )
    parser.add_argument(
        "--output_dir",
        default="data/finetune_data/",
        help="Where to save the generated datasets."
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    q_out = os.path.join(args.output_dir, "question_dataset.jsonl")
    a_out = os.path.join(args.output_dir, "answer_dataset.jsonl")
    # c_out = os.path.join(args.output_dir, "category_dataset.jsonl")  # category disabled

    # Prompt templates (multi‑abstract ready)
    question_template = (
        "You are a creative research assistant who generates thoughtful and meaningful questions "
        "from scientific abstracts. Only use the abstracts provided below. Do not invent or assume information. "
        "Craft a specific, probing, and original question that arises strictly from these abstracts. "
        "Do NOT generate multiple-choice, yes/no, true/false, or simple factual questions.\n\n"
        "Here are the abstracts:\n{abstract_block}\n\nQuestion:"
    )

    answer_template = (
        "You are an expert research assistant. Below are one or more scientific abstracts followed by a question. "
        "Provide a concise and insightful answer based only on the provided abstracts. "
        "If the abstracts do not contain enough information, respond with 'N/A'.\n\n"
        "Here are the abstracts:\n{abstract_block}\n\nQuestion: {question}\n\nAnswer:"
    )

    # ------------------------------------------------------------
    # Load combined JSONL
    # ------------------------------------------------------------
    with jsonlines.open(args.input, mode="r") as reader, \
         jsonlines.open(q_out, mode="w") as qwriter, \
         jsonlines.open(a_out, mode="w") as awriter:
         # jsonlines.open(c_out, mode="w") as cwriter  # category disabled

        for entry in reader:
            question = entry.get("question", "").strip()
            answer = entry.get("answer", "").strip()
            abstracts_dict = entry.get("abstract", {})
            # Join multiple abstracts with separator lines
            abstract_block = "\n\n".join(abstracts_dict.values()).strip()

            # Question FT record
            qwriter.write({
                "prompt": question_template.format(abstract_block=abstract_block),
                "response": question
            })

            # Answer FT record
            awriter.write({
                "prompt": answer_template.format(
                    abstract_block=abstract_block,
                    question=question
                ),
                "response": answer
            })

            # --- Category part commented out for now ---
            # cat_prompt = ...
            # cwriter.write({"prompt": cat_prompt, "response": category})

    print("Finished creating fine‑tuning datasets:")
    print(f" - {q_out}\n - {a_out}")
    # print(f" - {c_out}  (commented out)")

if __name__ == "__main__":
    main()