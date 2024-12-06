import json
import re
import random
from tqdm import tqdm

# keywords to preserve in the text
KEYWORDS = ["telomere", "mortality", "influenza", "pneumonia", "length", "HR", "BMI"]

def mask_digits(text):
    """
    Replaces numeric values with placeholders for uniformity, preventing random insertions into these values.
    
    Args:
        text (str): The input text where digits will be masked.
    
    Returns:
        str: Text with numeric values replaced by placeholders.
    """
    # Replace various numeric formats with 'DIGIT'
    text = re.sub(r"\b\d+\.\d{1,4}\b", "DIGIT.DIGITDIGITDIGIT", text)  # decimal numbers like 0.054
    text = re.sub(r"\b\d+\b", "DIGIT", text)  # integer numbers
    text = re.sub(r"\b\d+%\b", "DIGIT%", text)  # percentages
    return text

def unmask_digits(text, original_text):
    """
    Restores the original numeric values into the text where 'DIGIT' placeholders were used.

    Args:
        text (str): The text with 'DIGIT' placeholders.
        original_text (str): The original text to retrieve original numbers.

    Returns:
        str: Text with original numeric values restored.
    """
    # Find all numeric values in the original text
    original_numbers = re.findall(r"\b\d+\.\d{1,4}\b|\b\d+\b|\b\d+%\b", original_text)
    # Replace each placeholder with the original numbers sequentially
    for number in original_numbers:
        text = text.replace("DIGIT.DIGITDIGITDIGIT", number, 1)
        text = text.replace("DIGIT%", number, 1)
        text = text.replace("DIGIT", number, 1)
    return text

def random_insertion_attack(text, num_insertions=1):
    """
    Inserts random words into the text while preserving 'DIGIT' placeholders.

    Args:
        text (str): The original text to augment with random insertions.
        num_insertions (int): Number of random insertions to make.
    
    Returns:
        str: Text with random words inserted, excluding numeric placeholders.
    """
    # Split the text into words
    words = text.split()
    random_words = ["notably", "significant", "associated", "suggests", "correlated", "remarkably"]
    
    # Insert words randomly while ensuring no interference with numeric placeholders
    for _ in range(num_insertions):
        insert_word = random.choice(random_words)
        insert_position = random.randint(0, len(words))
        
        # Only insert if the position does not contain or affect 'DIGIT' placeholders
        if not re.search(r'DIGIT', words[insert_position - 1] if insert_position > 0 else ''):
            words.insert(insert_position, insert_word)
    
    return " ".join(words)

def augment_text(text, num_insertions=3):
    """
    Masks digits, applies random insertion, and unmasks the digits in text.

    Args:
        text (str): The original text.
        num_insertions (int): Number of random insertions to make.
    
    Returns:
        str: Augmented text with original digits restored.
    """
    masked_text = mask_digits(text)
    augmented_text = random_insertion_attack(masked_text, num_insertions)
    final_text = unmask_digits(augmented_text, text)
    return final_text

def augment_jsonl(input_file, output_file, num_variants=3):
    """
    Reads a JSONL file, applies augmentation on each entry, and writes the data to a new JSONL file.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to save the augmented JSONL data.
        num_variants (int): Number of augmented variants to generate for each entry.
    """
    with open(input_file, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    
    augmented_data = []
    for entry in tqdm(data, desc="Augmenting data"):
        question = entry["question"]
        answer = entry["answer"]
        pmid = entry["pmid"]
        
        # Generate augmented versions for question and answer
        augmented_questions = [augment_text(question, num_insertions=3) for _ in range(num_variants)]
        augmented_answers = [augment_text(answer, num_insertions=3) for _ in range(num_variants)]
        
        # Create new augmented entries with the same pmid
        for aq, aa in zip(augmented_questions, augmented_answers):
            augmented_data.append({
                "question": aq,
                "answer": aa,
                "pmid": pmid
            })
    
    # Write the augmented data to a new JSONL file
    with open(output_file, "w") as f:
        for item in augmented_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Augmentation complete! Augmented data saved to {output_file}")

if __name__ == "__main__":
    input_file = "./utils/submitQA/qa_database.jsonl"  # Input JSONL file with original data
    output_file = "./data/output/insertion_attack.jsonl"  # Output JSONL file for augmented data
    num_variants = 5  # Number of random insertion variants to generate for each entry
    augment_jsonl(input_file, output_file, num_variants=num_variants)
