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
                    "You are a creative research assistant who generates thoughtful and meaningful questions from one or more scientific abstracts. "
                    "Only use the abstracts provided below. Do not invent or assume information. "
                    "Craft a specific, probing, and original question that arises strictly from these abstracts. "
                    "Do NOT generate multiple-choice, yes/no, true/false, or simple factual questions.\n\n"
                    f"Here are the abstracts:\n{abstract}\n\n"
                    f"{anchor_q}"
                )
                question_config = {
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "top_k": 50
                }
                question_text = model_question.generate(prompt_question, question_config)
                question_text = extract_after_anchor(question_text, anchor_q)

                # Step 2: Generate Answer
                anchor_a = "Answer:"
                prompt_answer = (
                    "You are an expert research assistant. Below, you have one or more scientific abstracts followed by a question. "
                    "Provide a concise and insightful answer based only on the provided abstracts. If the abstracts do not contain enough information, respond with 'N/A'. "
                    "Do not invent or assume facts beyond what the abstracts state.\n\n"
                    f"Here are the abstracts:\n{abstract}\n\n"
                    f"Question: {question_text}\n\n"
                    f"{anchor_a}"
                )
                answer_config = {
                    "max_new_tokens": 100,
                    "temperature": 0.6,
                    "top_k": 40
                }
                answer_text = model_answer.generate(prompt_answer, answer_config)
                answer_text = extract_after_anchor(answer_text, anchor_a)

                # --- RE-EVALUATION IF ANSWER CONTAINS 'N/A' ---
                while "N/A" not in answer_text:
                    anchor_re = "Revised answer:"
                    reevaluation_prompt = (
                        "It appears your answer may be incomplete or marked as 'N/A'. "
                        "Please re-check the abstracts carefully to see if there's any evidence to provide a more informative answer. "
                        "If there's truly no information in the abstracts to answer, confirm with 'N/A'.\n\n"
                        "You are an expert research assistant. Below, you have one or more scientific abstracts followed by a question. "
                        "Provide a concise and insightful answer based only on the provided abstracts. "
                        "Do not invent or assume facts beyond what the abstracts state.\n\n"
                        f"Here are the abstracts:\n{abstract}\n\n"
                        f"Question: {question_text}\n\n"
                        f"{anchor_re}"
                    )
                    reeval_config = {
                        "max_new_tokens": 100,
                        "temperature": 0.6,
                        "top_k": 40
                    }
                    revised_answer_text = model_answer.generate(reevaluation_prompt, reeval_config)
                    revised_answer_text = extract_after_anchor(revised_answer_text, anchor_re).strip()

                # If the second attempt does not contain 'N/A', we assume it's a better answer
                answer_text = revised_answer_text

                # Step 3: Categorization
                anchor_c = "Category (choose only from method, knowledge, discussion):"
                categorization_prompt = (
                    "You are a scientific reviewer. Given the following Q&A pair, classify the type of insight provided as one of: method, knowledge, or discussion. "
                    "Choose the category that best fits the content of the answer. "
                    "If the answer does not fit any of these categories, respond with 'N/A'.\n\n"
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
