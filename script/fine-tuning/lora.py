import os
from transformers import Trainer, TrainingArguments, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import torch
from src.model import ItriModel
from src.utils import load_and_format_jsonl
from src.conf import conf as conf


def load_itrimodel(model_name, device="cuda"):
    return ItriModel(model_name=model_name, device=device)

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Assign eos_token as the pad token
    return tokenizer

def load_small_dataset(file_path, tokenizer, max_length=512):
    """
    Load the small dataset (4 examples) and tokenize for training.
    """
    data = load_and_format_jsonl(file_path)
    raw_data = [{"question": entry.split("\n")[0].split(": ")[1],
                 "context": entry.split("\n")[1].split(": ")[1],
                 "answer": entry.split("\n")[2].split(": ")[1]}
                for entry in data.split("\n\n") if entry.strip()]

    dataset = Dataset.from_dict(raw_data)

    def preprocess_function(examples):
        inputs = ["Question: " + q + " Context: " + c for q, c in zip(examples["question"], examples["context"])]
        targets = examples["answer"]
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)

def configure_lora():
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=4,  # Reduce rank dimension for smaller adaptations
        lora_alpha=8,
        lora_dropout=0.1
    )

def fine_tune_small_itrimodel(model, tokenizer, dataset, output_dir, save_dir):
    """
    Fine-tune the ItriModel with a very small dataset.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",               # Skip evaluation for small data
        learning_rate=5e-5,               # Higher learning rate for few examples
        per_device_train_batch_size=1,    # Batch size of 1
        num_train_epochs=1,               # Only one epoch
        weight_decay=0.0,                 # No regularization
        logging_dir='./logs',             # Logging directory
        save_total_limit=1,               # Save only one checkpoint
        fp16=True                         # Enable mixed precision
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save the fine-tuned model
    model.save_model(save_dir)

if __name__ == "__main__":
    # Paths
    model_name = "meta-llama/Llama-3.2-1B"  # Use smaller model
    qa_jsonl_path = "./utils/submitQA/qa_abstract.jsonl"
    output_dir = conf.results_path
    save_dir = conf.model_path

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = load_itrimodel(model_name, device=device)
    tokenizer = load_tokenizer(model_name)

    # Apply LoRA
    peft_config = configure_lora()
    model.model = get_peft_model(model.model, peft_config)

    # Load and preprocess the small dataset
    dataset = load_small_dataset(qa_jsonl_path, tokenizer)

    # Fine-tune the model
    fine_tune_small_itrimodel(model, tokenizer, dataset, output_dir, save_dir)
