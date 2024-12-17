from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from src.utils import load_jsonl_as_dict
import src.conf as conf
from prompt.prompt_manager import PromptManager
from peft import get_peft_model, LoraConfig, TaskType
import torch


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


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


def configure_lora():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=8,
        lora_dropout=0.1
    )


def fine_tune_model(model, tokenizer, train_dataset, eval_dataset, output_dir, save_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,  # Reduce evaluation batch size
        gradient_accumulation_steps=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=2,
        save_steps=500,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2  # Use multiple workers to reduce memory spikes
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=None,  # Custom metrics can be added if needed
    )

    trainer.train()

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    train_jsonl_path = "data/QA_data/train.jsonl"
    eval_jsonl_path = "./data/QA_data/eval.jsonl"
    output_dir = conf.results_path
    save_dir = conf.model_path
    template_dir = conf.template_path

    # Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = load_tokenizer(model_name)

    # Resize model embeddings if tokenizer was modified
    model.resize_token_embeddings(len(tokenizer))

    # Apply LoRA
    peft_config = configure_lora()
    model = get_peft_model(model, peft_config)

    # Load PromptManager
    prompt_manager = PromptManager()

    # Prepare datasets
    train_dataset = prepare_dataset(train_jsonl_path, tokenizer, prompt_manager)
    eval_dataset = prepare_dataset(eval_jsonl_path, tokenizer, prompt_manager)

    # Fine-tune the model
    fine_tune_model(model, tokenizer, train_dataset, eval_dataset, output_dir, save_dir)
