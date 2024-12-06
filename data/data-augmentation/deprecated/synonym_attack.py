import json
from tqdm import tqdm
import textattack
from textattack.augmentation import Augmenter
from textattack.transformations import WordSwapWordNet
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification

def synonym_replacement_attack(text, num_variants=1):
    """
    Generates augmented versions of the input text by replacing words with synonyms.
    
    Args:
        text (str): The original text to augment.
        num_variants (int): Number of augmented variants to generate.
    
    Returns:
        List[str]: A list of augmented variants of the input text.
    """
    # Augmenter with WordSwapWordNet transformation
    transformation = WordSwapWordNet()
    constraints = [RepeatModification(), StopwordModification()]
    augmenter = Augmenter(transformation=transformation, constraints=constraints, transformations_per_example=num_variants)

    return augmenter.augment(text)

def augment_jsonl(input_file, output_file, num_variants=3):
    """
    Reads a JSONL file, applies synonym replacement attack on each entry, and writes the augmented data to a new JSONL file.
    
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
        augmented_questions = synonym_replacement_attack(question, num_variants=num_variants)
        augmented_answers = synonym_replacement_attack(answer, num_variants=num_variants)
        
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
    output_file = "./data/output/synonym_attack.jsonl"  # Output JSONL file for augmented data
    augment_jsonl(input_file, output_file, num_variants=5)
