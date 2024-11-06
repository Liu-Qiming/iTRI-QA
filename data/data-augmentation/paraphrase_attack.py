import json
import re
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "ramsrigouthamg/t5_paraphraser"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# fixed keywords to be masked
KEYWORDS = ["telomere", "mortality", "influenza", "pneumonia", "length", "HR", "BMI"]

# Function to mask fixed keywords and digits with unique placeholders
def mask_important_terms(text):
    keyword_map = {}
    digit_map = {}
    masked_text = text

    # Mask keywords with unique placeholders
    for i, keyword in enumerate(KEYWORDS):
        placeholder = f"KEYWORD_{i}_POS"
        keyword_map[placeholder] = keyword
        masked_text = re.sub(r'\b{}\b'.format(re.escape(keyword)), placeholder, masked_text)
    
    # Mask digits with unique placeholders
    digits = re.findall(r'\b\d+\.?\d*\b', masked_text)
    for i, digit in enumerate(digits):
        placeholder = f"DIGIT_{i}_POS"
        digit_map[placeholder] = digit
        masked_text = re.sub(r'\b{}\b'.format(re.escape(digit)), placeholder, masked_text)

    return masked_text, keyword_map, digit_map

# Function to unmask keywords and digits after paraphrasing
def unmask_important_terms(paraphrased_text, keyword_map, digit_map):
    # Replace each unique placeholder with its corresponding original term
    for placeholder, keyword in keyword_map.items():
        paraphrased_text = paraphrased_text.replace(placeholder, keyword)
    for placeholder, digit in digit_map.items():
        paraphrased_text = paraphrased_text.replace(placeholder, digit)
    return paraphrased_text

# Paraphrase function with masking and unmasking
def paraphrase_with_preserved_terms(text, num_variants=1, max_length=128):
    masked_text, keyword_map, digit_map = mask_important_terms(text)
    
    paraphrased_texts = []
    for _ in range(num_variants):
        input_text = f"paraphrase: {masked_text} </s>"
        inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True)

        # Generate paraphrase
        outputs = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            do_sample=True,  
            top_k=50,        
            top_p=0.9,       
            temperature=0.8, 
            num_return_sequences=1
        )
        
        # Decode the paraphrased text
        paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Unmask important terms
        unmasked_text = unmask_important_terms(paraphrased_text, keyword_map, digit_map)

        paraphrased_texts.append(unmasked_text)

    return paraphrased_texts

# Function to apply paraphrasing to JSONL data with preserved terms
def augment_jsonl(input_file, output_file, num_variants=3):
    with open(input_file, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    
    augmented_data = []
    for entry in tqdm(data, desc="Paraphrasing data"):
        question = entry["question"]
        answer = entry["answer"]
        pmid = entry["pmid"]
        
        # Generate paraphrased versions for question and answer with term preservation
        paraphrased_questions = paraphrase_with_preserved_terms(question, num_variants=num_variants)
        paraphrased_answers = paraphrase_with_preserved_terms(answer, num_variants=num_variants)
        
        # Create new augmented entries with the same pmid
        for pq, pa in zip(paraphrased_questions, paraphrased_answers):
            augmented_data.append({
                "question": pq,
                "answer": pa,
                "pmid": pmid
            })
    
    # Write the augmented data to a new JSONL file
    with open(output_file, "w") as f:
        for item in augmented_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Augmentation complete! Augmented data saved to {output_file}")

if __name__ == "__main__":
    input_file = "./utils/submitQA/qa_database.jsonl"  # Input JSONL file with original data
    output_file = "./data/output/paraphrase_attack.jsonl"  # Output JSONL file for augmented data
    num_variants = 5  # Number of random deletion variants to generate for each entry
    augment_jsonl(input_file, output_file, num_variants=num_variants)
