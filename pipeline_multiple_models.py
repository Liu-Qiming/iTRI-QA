import argparse
import torch
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc
import re
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_question_name = "meta-llama/Llama-3.2-3B"
    model_answer_name = "meta-llama/Llama-3.2-3B"
    model_category_name = "meta-llama/Llama-3.2-3B"

    tokenizer_question = AutoTokenizer.from_pretrained(model_question_name)
    model_question = AutoModelForCausalLM.from_pretrained(model_question_name).to(device)

    tokenizer_answer = AutoTokenizer.from_pretrained(model_answer_name)
    model_answer = AutoModelForCausalLM.from_pretrained(model_answer_name).to(device)

    tokenizer_category = AutoTokenizer.from_pretrained(model_category_name)
    model_category = AutoModelForCausalLM.from_pretrained(model_category_name).to(device)

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
                    "Read the abstract provided below and use your expert judgment to select a distinctive angle or critical insight to explore."
                    " Avoid generic inquiries and instead craft a specific, probing, and original question.\n\n"
                    f"Abstract:\n{abstract}\n\n{anchor_q}"
                )
                inputs_question = tokenizer_question(prompt_question, return_tensors="pt").to(device)
                question_output = model_question.generate(
                    inputs_question.input_ids,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=50
                )
                question_raw = tokenizer_question.decode(question_output[0], skip_special_tokens=True)
                question_text = extract_after_anchor(question_raw, anchor_q)

                # Step 2: Generate Answer
                anchor_a = "Answer:"
                prompt_answer = (
                    "You are an expert research assistant. Based on the scientific abstract and the question below, provide a concise and insightful answer that reflects the key points in the abstract.\n\n"
                    f"Abstract:\n{abstract}\n\n"
                    f"Question: {question_text}\n\n{anchor_a}"
                )
                inputs_answer = tokenizer_answer(prompt_answer, return_tensors="pt").to(device)
                answer_output = model_answer.generate(
                    inputs_answer.input_ids,
                    max_new_tokens=150,
                    temperature=0.6,
                    top_k=40
                )
                answer_raw = tokenizer_answer.decode(answer_output[0], skip_special_tokens=True)
                answer_text = extract_after_anchor(answer_raw, anchor_a)

                # Step 3: Categorization
                anchor_c = "Category (choose only from method, knowledge, discussion):"
                categorization_prompt = (
                    "You are a scientific reviewer. Given the following Q&A pair, classify the type of insight provided as one of: method, knowledge, or discussion.\n\n"
                    f"Question: {question_text}\n"
                    f"Answer: {answer_text}\n\n{anchor_c}"
                )
                inputs_category = tokenizer_category(categorization_prompt, return_tensors="pt").to(device)
                category_output = model_category.generate(
                    inputs_category.input_ids,
                    max_new_tokens=1,
                    temperature=0.1,
                    top_k=10
                )
                category_raw = tokenizer_category.decode(category_output[0], skip_special_tokens=True)
                category = extract_after_anchor(category_raw, anchor_c)

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