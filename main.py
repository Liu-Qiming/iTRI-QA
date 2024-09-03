import argparse
import json
from peft import LoraConfig

from src.model import ItriModel  
from src.utils import *
import src.conf as conf

def main():
    parser = argparse.ArgumentParser(description="Use ItriModel for Medical Q&A")
    parser.add_argument("--model_name", type=str, default=conf.model_name, help="Name of the model to use")
    args = parser.parse_args()

    model = ItriModel(model_name=args.model_name)

    try:
        queries = load_json_data(conf.query_path)["query"]
        context = [process_csv_data(conf.context_path, 5)]
    except FileNotFoundError as e:
        print(e)
        return
    queries_contexts = list(zip(queries, context))

    print("Generating answers for the batch of queries and contexts...")
    answers = model.batch_generate(queries_contexts)

    for (query, context), answer in zip(queries_contexts, answers):
        print(f"Question: {query}")
        print(f"Context: {context}\n\n")
        print("---------------------------------")
        print(f"Answer: {answer}\n")

    model.save_model(base_save_path="./models")

if __name__ == "__main__":
    main()

