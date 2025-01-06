import argparse
import torch
import jsonlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.model import ItriModel
from prompt.prompt_manager import PromptManager
import src.conf as conf
from tqdm import tqdm

from utils.load_abstract_db.file_readers import read_output_yaml_file


def main():
    parser = argparse.ArgumentParser(description="Use ItriModel for Medical Q&A")
    parser.add_argument("--model_name", type=str, default=conf.model_name, help="Name of the model to use")
    parser.add_argument("--yaml_path", type=str, default="utils/load_abstract_db/output.yaml", help="Path to the YAML file")
    parser.add_argument("--output_path", type=str, default="output.jsonl", help="Path to the output JSONL file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the Q&A model
    model_qa = ItriModel("meta-llama/Llama-3.2-3B", conf.adapter_path)

    # Initialize the categorization model
    category_model_name = "meta-llama/Llama-3.2-3B"  # Using the same model for categorization
    tokenizer = AutoTokenizer.from_pretrained(category_model_name)
    category_model = AutoModelForCausalLM.from_pretrained(category_model_name).to(device)

    prompt_manager = PromptManager()

    try:
        yaml_data = read_output_yaml_file(args.yaml_path)

        results = []

        for item in tqdm(yaml_data):
            doi = item.get("doi", "unknown")
            abstract = item.get("abstract", "")

            # Step 1: Generate Q&A set
            print(f"Generating Q&A set for DOI: {doi}")
            prompt_qa = prompt_manager.render_prompt("llama3.2.j2", {"abstract": abstract})
            qa_set_raw = model_qa.generate(prompt_qa)

            qa_keyphrase = "# Your generated question and answer set:"
            qa_set = qa_set_raw.split(qa_keyphrase, 1)[1].strip() if qa_keyphrase in qa_set_raw else ""

            # Step 2: Categorize
            print(f"Categorizing Q&A set for DOI: {doi}")
            categorization_prompt = (
                f"Given the following Q&A set, classify it into one of the following categories:"
                f" method, knowledge, discussion.\n\n"
                f"Q&A Set: {qa_set}\n"
                f"Category (choose only from method, knowledge, discussion):"
            )

            inputs = tokenizer(categorization_prompt, return_tensors="pt").to(device)
            outputs = category_model.generate(
                inputs.input_ids,
                max_new_tokens=1,
                temperature=0.1
            )
            category_raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            category_keyphrase = "Category (choose only from method, knowledge, discussion):"
            category = category_raw.split(category_keyphrase, 1)[1].strip() if category_keyphrase in category_raw else ""

            # Store results
            results.append({
                "doi": doi,
                "QA": qa_set,
                "category": category,
            })

        # Write results to JSONL file
        print(f"Saving results to {args.output_path}")
        with jsonlines.open(args.output_path, mode='w') as writer:
            writer.write_all(results)

        print("Processing complete.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
