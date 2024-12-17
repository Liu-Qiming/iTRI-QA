import argparse
import torch
from src.model import ItriModel
from src.utils import *
from prompt.prompt_manager import PromptManager
import src.conf as conf
import torch


def main():
    parser = argparse.ArgumentParser(description="Use ItriModel for Medical Q&A")
    parser.add_argument("--model_name", type=str, default=conf.model_name, help="Name of the model to use")
    args = parser.parse_args()

    # Automatically default to 'cuda' if available, else fallback to 'cpu'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the model
    model = ItriModel(args.model_name, conf.adapter_path)

    # Initialize the PromptManager
    prompt_manager = PromptManager()

    try:
        # Load queries and context
        # queries = load_jsonl_as_dict(conf.QA_data_path)[0]["question"]
        context_list = load_jsonl_as_dict(conf.QA_data_path)

        # for ctx in context_list:
        #     # Render and generate answers
        #     print("Generating QA set based on the abstracts")
        #     # Render the prompt using PromptManager
        #     prompt = prompt_manager.render_prompt("llama3.2.j2", {"context": ctx["abstract"]})
        #     answer = model.generate(prompt)

        #     # Print results
        #     # print(f"Query: {query}")
        #     # print(f"Context: {context}\n")
        #     # print("---------------------------------")
        #     print(f"Answer: {answer}\n")
        #     print("---------------------------------")
        prompt = prompt_manager.render_prompt("llama3.2.j2", {"abstract": context_list[0]["abstract"]})
        answer = model.generate(prompt)
        print(answer)
        # Save the model after generation
        # model.save_model(base_save_path="./models")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
