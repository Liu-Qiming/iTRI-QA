import json
import random

def extract_random_n_examples(train_jsonl_path, destination_jsonl_path, N):
    """
    Extracts N random examples from a source JSONL file and saves them into a destination JSONL file.
    
    :param train_jsonl_path: Path to the source JSONL file.
    :param destination_jsonl_path: Path to the destination JSONL file.
    :param N: Number of examples to extract.
    """
    try:
        # Read all examples from the source JSONL file
        with open(train_jsonl_path, 'r') as infile:
            examples = [json.loads(line) for line in infile]

        # Check if N exceeds the available examples
        if N > len(examples):
            raise ValueError(f"Requested {N} examples, but only {len(examples)} are available.")

        # Randomly sample N examples
        random_examples = random.sample(examples, N)

        # Write the random examples to the destination JSONL file
        with open(destination_jsonl_path, 'w') as outfile:
            for example in random_examples:
                json.dump(example, outfile)
                outfile.write('\n')

        print(f"Successfully extracted {N} random examples to {destination_jsonl_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
train_jsonl_path = "data/QA_data/augmented/augmented_qa_abstract.jsonl"
destination_jsonl_path = "data/QA_data/eval.jsonl"
N = 2000  # Number of random examples to extract

extract_random_n_examples(train_jsonl_path, destination_jsonl_path, N)
