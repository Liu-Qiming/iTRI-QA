import json
import jsonlines
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to your existing database JSON/JSONL", default="utils/submit_QA_sample/qa_database_updated.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/finetune_data/", help="Where to save the generated datasets")
    args = parser.parse_args()

    # Make sure output dir exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare output file paths
    question_path = os.path.join(args.output_dir, "question_dataset.jsonl")
    answer_path = os.path.join(args.output_dir, "answer_dataset.jsonl")
    category_path = os.path.join(args.output_dir, "category_dataset.jsonl")

    # The EXACT prompts from your main script
    question_prompt_template = (
        "You are a creative research assistant who generates thoughtful and meaningful questions "
        "from scientific abstracts. Only use the abstract provided below. Do not invent or assume information. "
        "Craft a specific, probing, and original question based strictly on the abstract. "
        "Do NOT generate multiple-choice questions, Yes/No, True/False, Selection, or simple factual questions.\n\n"
        "Abstract:\n{abstract}\n\nQuestion:"
    )

    answer_prompt_template = (
        "You are an expert research assistant. Based on the scientific abstract and the question below, "
        "provide a concise and insightful answer. Base your response strictly on the content of the abstract. "
        "If the abstract does not contain enough information, respond with 'N/A'\n\n"
        "Abstract:\n{abstract}\n\nQuestion: {question}\n\nAnswer:"
    )

    category_prompt_template = (
        "You are a scientific reviewer. Given the following Q&A pair, classify the type of insight provided "
        "as one of: method, knowledge, or discussion. Choose the category that best fits the content of the answer.\n\n"
        "Question: {question}\nAnswer: {answer}\n\n"
        "Category (choose only from method, knowledge, discussion):"
    )

    # Load the database
    # We assume it's a JSONL, but if it's a single JSON array, adapt accordingly
    data_entries = []
    if args.input.endswith(".jsonl"):
        with jsonlines.open(args.input, "r") as reader:
            for obj in reader:
                data_entries.append(obj)
    else:
        # assume .json
        with open(args.input, "r", encoding="utf-8") as f:
            data_entries = json.load(f)

    # Create each dataset
    with jsonlines.open(question_path, mode="w") as q_writer, \
         jsonlines.open(answer_path, mode="w") as a_writer, \
         jsonlines.open(category_path, mode="w") as c_writer:

        for entry in data_entries:
            # Some fields: question, answer, category, abstract
            abs_text = entry.get("abstract", "")
            question_text = entry.get("question", "")
            answer_text = entry.get("answer", "")
            category = entry.get("category", "")

            # 1) Question data
            q_prompt = question_prompt_template.format(abstract=abs_text)
            # The "response" is the actual question from your DB
            q_writer.write({
                "prompt": q_prompt,
                "response": question_text
            })

            # 2) Answer data
            a_prompt = answer_prompt_template.format(
                abstract=abs_text,
                question=question_text
            )
            a_writer.write({
                "prompt": a_prompt,
                "response": answer_text
            })

            # 3) Category data
            c_prompt = category_prompt_template.format(
                question=question_text,
                answer=answer_text
            )
            c_writer.write({
                "prompt": c_prompt,
                "response": category
            })

    print("Finished creating fine-tuning datasets.")
    print(f" - {question_path}\n - {answer_path}\n - {category_path}")

if __name__ == "__main__":
    main()
