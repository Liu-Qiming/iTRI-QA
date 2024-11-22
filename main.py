import argparse
import torch
from jinja2 import Template

from src.model import ItriModel
from src.utils import *
import src.conf as conf


def main():
    parser = argparse.ArgumentParser(description="Use ItriModel for Medical Q&A")
    parser.add_argument("--model_name", type=str, default=conf.model_name, help="Name of the model to use")
    args = parser.parse_args()

    # Automatically default to 'cuda' if available, else fallback to 'cpu'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the model
    model = ItriModel(model_name=args.model_name, device=device)

    try:
        # Load queries and context
        queries = load_json_data(conf.query_path)["query"]
        context = process_csv_data(conf.context_path, 5)

        # Load and render the template
        with open("prompt/templates/llama3.2.j2", "r") as template_file:
            template = Template(template_file.read())
    except FileNotFoundError as e:
        print(e)
        return

    print("Generating...")
    for query in queries:
        # Render the prompt using Jinja2
        prompt = template.render(query=query, context=context)

        # Pass the rendered prompt to the model
        answer = model.generate(prompt)

        # print(f"Question: {query}")
        # print(f"Context: {context}\n")
        # print("---------------------------------")
        print(f"Answer: {answer}\n")

    # model.save_model(base_save_path="./models")


if __name__ == "__main__":
    main()
