from transformers import AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from transformers import LlamaForSequenceClassification
from datasets import load_dataset

# 1. Load LLaMA model and tokenizer
model_name = "huggyllama/llama-7b"  # Using LLaMA-7B as an example. Adjust this based on your model.
model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Configure LoRA parameters
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,  # Sequence classification task
    inference_mode=False,        # We're training, not just inference
    r=8,                         # Low-rank dimension (controls adaptation capacity)
    lora_alpha=16,               # Scaling factor for LoRA
    lora_dropout=0.1             # Dropout to prevent overfitting
)

# Apply LoRA to the LLaMA model
model = get_peft_model(model, peft_config)

# 3. Load dataset and preprocess (using IMDb dataset)
dataset = load_dataset("imdb")

# Assign the eos_token as the padding token
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset with padding enabled
def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True)

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Split dataset into train and test subsets
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # For fine-tuning, using a small subset
test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# 4. Define training arguments
training_args = TrainingArguments(
    output_dir='./results_llama_lora',  # Directory for saving results
    eval_strategy="epoch",              # Updated: Evaluation strategy per epoch
    learning_rate=2e-5,                 # Learning rate for fine-tuning
    per_device_train_batch_size=8,      # Batch size per device for training
    per_device_eval_batch_size=8,       # Batch size per device for evaluation
    num_train_epochs=3,                 # Number of fine-tuning epochs
    weight_decay=0.01,                  # Weight decay to regularize the model
    logging_dir='./logs_llama_lora',    # Directory for logging
    save_total_limit=2,                 # Limit the number of saved model checkpoints
)

# 5. Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# 6. Fine-tune the model with LoRA
trainer.train()

# 7. Evaluate the fine-tuned LLaMA model
results = trainer.evaluate()
print(f"Evaluation Results: {results}")

# 8. Save the fine-tuned model
model.save_pretrained("./fine_tuned_llama_lora_model")
tokenizer.save_pretrained("./fine_tuned_llama_lora_tokenizer")

print("Fine-tuned LLaMA model and tokenizer saved successfully!")
