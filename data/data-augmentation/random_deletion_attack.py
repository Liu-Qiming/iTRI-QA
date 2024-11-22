import json
import random
import re
from tqdm import tqdm

# keywords to preserve
KEYWORDS = {"telomere", "mortality", "influenza", "pneumonia", "length", "HR", "BMI"}

def find_important_numbers(text):
    """
    Extracts important numbers from the text using regular expressions.
    
    Args:
        text (str): The input text.
    
    Returns:
        Set[str]: A set of strings containing important numbers found in the text.
    """
    # Regular expression to match numbers (e.g., 0.54, 12%, 3.14e10, 25)
    number_pattern = r"\b\d+\.?\d*%?\b|\b\d+e[-+]?\d+\b"
    return set(re.findall(number_pattern, text))

def random_deletion_attack(text, deletion_probability=0.2):
    """
    Generates an augmented version of the input text by randomly deleting words,
    while preserving keywords and important numbers.
    
    Args:
        text (str): The original text to augment.
        deletion_probability (float): Probability of deleting each word.
    
    Returns:
        str: An augmented version of the input text with some words removed.
    """
    # Combine keywords and important numbers
    important_words = KEYWORDS | find_important_numbers(text)

    words = text.split()
    if len(words) == 1:  # If the text is a single word, return it as is
        return text

    # Delete words randomly based on the specified probability,
    # while preserving important words
    augmented_words = [
        word for word in words 
        if word in important_words or random.random() > deletion_probability
    ]
    
    # Ensure at least one word is kept if all were deleted
    if not augmented_words:
        augmented_words = [random.choice(words)]
    
    return " ".join(augmented_words)

def augment_jsonl(input_file, output_file, num_variants=3, deletion_probability=0.2):
    """
    Reads a JSONL file, applies random deletion attack on each entry, 
    and writes the augmented data to a new JSONL file.
    
    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to save the augmented JSONL data.
        num_variants (int): Number of augmented variants to generate for each entry.
        deletion_probability (float): Probability of deleting each word in the text.
    """
    with open(input_file, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    
    augmented_data = []
    for entry in tqdm(data, desc="Augmenting data"):
        question = entry["question"]
        answer = entry["answer"]
        pmid = entry["pmid"]
        
        # Generate augmented versions for question and answer
        augmented_questions = [random_deletion_attack(question, deletion_probability) for _ in range(num_variants)]
        augmented_answers = [random_deletion_attack(answer, deletion_probability) for _ in range(num_variants)]
        
        # Create new augmented entries with the same pmid
        for dq, da in zip(augmented_questions, augmented_answers):
            augmented_data.append({
                "question": dq,
                "answer": da,
                "pmid": pmid
            })
    
    # Write the augmented data to a new JSONL file
    with open(output_file, "w") as f:
        for item in augmented_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Augmentation complete! Augmented data saved to {output_file}")

if __name__ == "__main__":
    input_file = "./utils/submitQA/qa_database.jsonl"  # Input JSONL file with original data
    output_file = "./data/output/deletion_attack.jsonl"  # Output JSONL file for augmented data
    num_variants = 5  # Number of random deletion variants to generate for each entry
    deletion_probability = 0.1  # Probability of deleting each word in the text
    augment_jsonl(input_file, output_file, num_variants=num_variants, deletion_probability=deletion_probability)
