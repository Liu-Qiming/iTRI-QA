import argparse
import torch
import jsonlines
from tqdm import tqdm
import random

from src.model import ItriModel
import src.conf as conf
from utils.load_abstract_db.file_readers import read_output_yaml_file


def extract_after_anchor(output_text, anchor):
    if anchor in output_text:
        return output_text.split(anchor, 1)[1].strip()
    return output_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Optimized ItriModel Pipeline for Medical Q&A and Categorization")
    parser.add_argument("--model_name", type=str, default=conf.model_name, help="Name of the model to use")
    parser.add_argument("--yaml_path", type=str, default="utils/load_abstract_db/output.yaml", help="Path to the YAML file")
    parser.add_argument("--output_path", type=str, default="output.jsonl", help="Path to the output JSONL file")
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to process")
    args = parser.parse_args()

    # Initialize ItriModel for all tasks
    model_question = ItriModel("meta-llama/Llama-3.2-3B")
    model_answer = ItriModel("tiiuae/Falcon3-3B-Instruct")
    model_category = ItriModel("meta-llama/Llama-3.2-3B")
    try:
        yaml_data = read_output_yaml_file(args.yaml_path)
        random.shuffle(yaml_data)
        if args.num_examples:
            yaml_data = yaml_data[:args.num_examples]

        with jsonlines.open(args.output_path, mode='w') as writer:
            for item in tqdm(yaml_data, desc="Processing abstracts"):
                doi = item.get("doi", "unknown")
                abstract = item.get("abstract", "")

                # Step 1: Generate Question
                anchor_q = "Question:"
                prompt_question = (
                    "You are a creative research assistant who generates thoughtful and meaningful questions from scientific abstracts. "
                    "Only use the abstract provided below. Do not invent or assume information. "
                    "Craft a specific, probing, and original question based strictly on the abstract. "
                    "Do NOT generate multiple-choice questions, Yes/No, True/False, Selection, or simple factual questions.\n\n"
                    f"Abstract:\n{abstract}\n\n{anchor_q}"
                )
                question_config = {
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "top_k": 50
                }
                question_text = model_question.generate(prompt_question, question_config)
                question_text = extract_after_anchor(question_text, anchor_q)

                # Step 2: Generate Answer
                anchor_a = "Answer:"
                prompt_answer = (
                    "You are an expert research assistant. Based on the scientific abstract and the question below, provide a concise and insightful answer. "
                    "Base your response strictly on the content of the abstract. If the abstract does not contain enough information, respond with 'N/A'\n\n"
                    f"Abstract:\n{abstract}\n\n"
                    f"Question: {question_text}\n\n{anchor_a}"
                )
                answer_config = {
                    "max_new_tokens": 150,
                    "temperature": 0.6,
                    "top_k": 40
                }
                answer_text = model_answer.generate(prompt_answer, answer_config)
                answer_text = extract_after_anchor(answer_text, anchor_a)

                # Step 3: Categorization
                anchor_c = "Category (choose only from method, knowledge, discussion):"
                categorization_prompt = (
                    "You are a scientific reviewer. Given the following Q&A pair, classify the type of insight provided as one of: method, knowledge, or discussion. "
                    "Choose the category that best fits the content of the answer.\n\n"
                    f"Question: {question_text}\n"
                    f"Answer: {answer_text}\n\n{anchor_c}"
                )
                category_config = {
                    "max_new_tokens": 1,
                    "temperature": 0.1,
                    "top_k": 10
                }
                category = model_category.generate(categorization_prompt, category_config)
                category = extract_after_anchor(category, anchor_c)

                writer.write({
                    "doi": doi,
                    "question": question_text,
                    "answer": answer_text,
                    "category": category,
                })

        print("Processing complete. Results saved.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()