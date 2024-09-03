from peft import LoraConfig

# Model config
model_config = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 50,
    "max_new_tokens": 120,
    "do_sample": True,
    "num_beams": 4,
    "early_stopping": True,
}

# Tokenizer config
tokenizer_config = {
    "return_tensors": "pt",
    "padding": True,
    "truncation": True,
    "max_length": 4096
}
# Default configuration for a Q&A task using LoRA
lora_config = LoraConfig(
    r=16,
    target_modules=["q", "v"],
    lora_alpha=32,
    lora_dropout=0.1,
    bias="lora_only"
)

model_name = "meta-llama/Llama-3.2-1B"

# TODO: Model Path
model_path = "models/"

# TODO: Adapter Path
adapter_path = "models/"

# TODO: Template Path
template_path = "prompt/templates/"

# Context Path
context_path = "data/context/Cholesterol - Low - Density Lipoprotein (LDL) & Triglycerides:2015-2016.csv"

# Query Path
query_path = "data/query/query.json"
