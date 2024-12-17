from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
import os
from abc import ABC, abstractmethod


class BaseLLMModel(ABC):
    def __init__(self, model_name: str):
        """Initialize the model with the given name."""
        self.model_name = model_name

    @abstractmethod
    def load_tokenizer(self, model_name: str):
        """Load the tokenizer for any HuggingFace LLM."""
        pass

    @abstractmethod
    def load_model(self, model_name: str):
        """Load any HuggingFace LLM model."""
        pass

    @abstractmethod
    def apply_perf_optimizations(self):
        """Apply performance optimizations."""
        pass

    @abstractmethod
    def generate(self, input_text: str):
        """Generate text based on input."""
        pass

    @abstractmethod
    def save_model(self, base_save_path: str):
        """Save the model and tokenizer with versioning."""
        pass


class ItriModel(BaseLLMModel):
    def __init__(self, model_name: str, adapter_path: str = None):
        """
        Initialize the model, tokenizer, and optionally apply an adapter.

        Args:
            model_name (str): Name of the base model.
            adapter_path (str, optional): Path to the adapter. Defaults to None.
            device (str): Device to load the model on. Defaults to "cuda" if available.
        """
        super().__init__(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = self.load_tokenizer(model_name)
        self.model = self.load_model(model_name)

        if adapter_path:
            self.apply_adapter(adapter_path)

        self.apply_perf_optimizations()

    def load_tokenizer(self, model_name: str):
        """Load the tokenizer and add a padding token if needed."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer

    def load_model(self, model_name: str):
        """
        Load the model with memory-efficient configurations using BitsAndBytesConfig.
        """
        # Patch for SCB attribute error in bitsandbytes
        if not hasattr(torch.nn.Parameter, "SCB"):
            setattr(torch.nn.Parameter, "SCB", None)

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True  # Offload unsupported layers to CPU if needed
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True  # Enable custom model implementations
        )

        # Resize token embeddings if tokenizer is updated
        if len(self.tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(self.tokenizer))

        return model

    def apply_adapter(self, adapter_path: str):
        """
        Apply an adapter to the loaded model.

        Args:
            adapter_path (str): Path to the adapter to be applied.
        """
        print(f"Loading adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def apply_perf_optimizations(self):
        """Enable performance optimizations to reduce memory usage."""
        self.model.gradient_checkpointing_enable()  # Save memory during training/inference

    def generate(self, prompt: str):
        """
        Generate an answer using the provided prompt.

        Args:
            prompt (str): The rendered prompt.

        Returns:
            str: The generated answer.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        )
        batch = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.amp.autocast(device_type="cuda" if self.device == "cuda" else "cpu", enabled=self.device == "cuda"):
            output = self.model.generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=200,  # Increased from 128 to allow for longer responses
                do_sample=True,  # Keep sampling for diversity
                num_beams=5,  # Increased from 2 to explore a broader search space
                temperature=0.7,  # Added to control randomness
                top_k=50,  # Focus on the top 50 tokens
                top_p=0.9,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return decoded_output.strip()

    def save_model(self, base_save_path: str):
        """Save the model and tokenizer with versioning."""
        model_dir = os.path.join(base_save_path, self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Determine the latest version number
        existing_versions = [d for d in os.listdir(model_dir) if d.startswith('v')]
        new_version = f"v{max([int(d[1:]) for d in existing_versions], default=0) + 1}" if existing_versions else "v1"

        # Create the directory for the new version
        version_dir = os.path.join(model_dir, new_version)
        os.makedirs(version_dir, exist_ok=True)

        # Save the model and tokenizer
        self.model.save_pretrained(version_dir)
        self.tokenizer.save_pretrained(version_dir)
        print(f"Model and tokenizer saved to {version_dir}")
