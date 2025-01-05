import os
import random
import json
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, IntervalStrategy
from datasets import Dataset
from src.model import ItriModel
from src.utils import load_jsonl_as_dict
from prompt.prompt_manager import PromptManager
from peft import get_peft_model, LoraConfig, TaskType
import torch
import numpy as np


# Constants
data_path = "utils/submit_QA_sample/qa_database_updated.jsonl"
eval_path = "data/QA_data/eval.jsonl"
result_path = "models/experiment/result.txt"
output_dir = "models/experiment"

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

# Step 1: Load and sample training data
def load_sampled_data(file_path, sample_sizes):
    data = load_jsonl_as_dict(file_path)
    sampled_datasets = {}
    for size in sample_sizes:
        sampled_datasets[size] = random.sample(data, min(size, len(data)))
    return sampled_datasets

# Step 2: Prepare dataset for fine-tuning
def prepare_dataset(file_path, tokenizer, prompt_manager, max_length=512):
    data = load_jsonl_as_dict(file_path)
    raw_data = [{"context": entry["abstract"], "question": entry["question"], "answer": entry["answer"]} for entry in data]
    dataset = Dataset.from_list(raw_data)

    def preprocess_function(examples):
        inputs = [
            prompt_manager.render_prompt("llama3.2.j2", {"context": context}) for context in examples["context"]
        ]
        targets = [
            f'{{"question": "{question}", "answer": "{answer}"}}'
            for question, answer in zip(examples["question"], examples["answer"])
        ]
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess_function, batched=True)

# Step 3: Configure LoRA for the model
def configure_lora():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Increase LoRA rank for better task-specific learning
        lora_alpha=32,  # Higher scaling factor for small datasets
        lora_dropout=0.1  # Slightly higher dropout to prevent overfitting
    )

# Step 4: Fine-tune the model
def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, current_output_dir, save_dir, num_epochs):
    training_args = TrainingArguments(
        output_dir=current_output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        evaluation_strategy=IntervalStrategy.NO,
        num_train_epochs=num_epochs,
        weight_decay=0.1,
        logging_dir="out.log",
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=None
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return trainer


# Step 5: Evaluate model and log results
def evaluate_and_log_results(trainer, eval_dataset, tokenizer, result_path, qa_size):
    predictions = trainer.predict(eval_dataset)
    decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

    accuracy = sum(1 for pred, label in zip(decoded_preds, decoded_labels) if pred.strip() == label.strip()) / len(decoded_labels)

    # Append result
    with open(result_path, "a") as result_file:
        result_file.write(f"QA Size: {qa_size}, Accuracy: {accuracy:.4f}\n")

def main():
    model_names = ["meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-3B"]
    sample_sizes = [3, 10, 25, 50]
    num_epochs = 500
    prompt_manager = PromptManager()

    full_train_data = load_jsonl_as_dict(data_path)

    for model_name in model_names:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = load_tokenizer(model_name)
        eval_dataset = prepare_dataset(eval_path, tokenizer, prompt_manager)
        peft_config = configure_lora()
        model = get_peft_model(model, peft_config)
        model.resize_token_embeddings(len(tokenizer))

        for qa_size in sample_sizes:
            sampled_data = random.sample(full_train_data, qa_size) if len(full_train_data) >= qa_size else full_train_data
            temp_jsonl_path = f"temp_sampled_data_{qa_size}.jsonl"
            with open(temp_jsonl_path, 'w') as f:
                for item in sampled_data:
                    f.write(json.dumps(item) + '\n')
            train_dataset = prepare_dataset(temp_jsonl_path, tokenizer, prompt_manager)
            experiment_dir = os.path.join(output_dir, f"{model_name.replace('/', '_')}_r16alpha32_QA{qa_size}")
            os.makedirs(experiment_dir, exist_ok=True)
            trainer = fine_tune_model(model, tokenizer, train_dataset, eval_dataset, output_dir, experiment_dir, num_epochs)
            os.remove(temp_jsonl_path)

            # Evaluate the model after training
            eval_result = trainer.evaluate(eval_dataset=eval_dataset)

            # Log the results
            with open(result_path, "a") as result_file:
                result_file.write(f"Model: {model_name}, QA Size: {qa_size}, Eval_Result: {eval_result}\n")

if __name__ == "__main__":
    main()