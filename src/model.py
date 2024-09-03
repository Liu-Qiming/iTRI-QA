from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from abc import ABC, abstractmethod
from peft import LoraConfig, get_peft_model
import src.conf as conf


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
        """Apply performance optimizations using a library like perf."""
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
    def __init__(self, model_name: str):
        """Initialize the model, tokenizer, and apply performance optimizations if needed."""
        super().__init__(model_name)
        self.tokenizer = self.load_tokenizer(model_name)
        self.model = self.load_model(model_name)
        self.apply_perf_optimizations()

    def load_tokenizer(self, model_name: str):
        """Load the tokenizer for any HuggingFace LLM and ensure it has a unique padding token."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Check if the tokenizer has a padding token, if not, add one
        if tokenizer.pad_token is None:
            # Add a unique [PAD] token if no padding token exists
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Ensure that pad_token_id is not the same as eos_token_id
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            raise ValueError("The `pad_token_id` should not be the same as `eos_token_id`. Please ensure a unique `pad_token_id`.")

        return tokenizer

    def load_model(self, model_name: str):
        """Load any HuggingFace LLM model and resize the token embeddings if needed."""
        model = AutoModelForCausalLM.from_pretrained(model_name)

        # Resize model embeddings if new tokens were added to the tokenizer
        if len(self.tokenizer) > model.config.vocab_size:
            model.resize_token_embeddings(len(self.tokenizer))

        return model

    def apply_perf_optimizations(self):
        """Apply performance optimizations (optional)."""
        print("Performance optimizations can be applied here if needed.")

    def generate(self, query: str, context: str):
        """Generate answers for the given question based on the context."""
        input_text = f"Question: {query}\nContext: {context}\nAnswer:"
        inputs = self.tokenizer(
            input_text,
            return_tensors=conf.tokenizer_config["tokenizer_config"],
            padding=conf.tokenizer_config["padding"],
            truncation=conf.tokenizer_config["truncation"],
            max_length=conf.tokenizer_config["max_length"]
        )

        # Pass attention mask to indicate which tokens should be attended to
        attention_mask = inputs['attention_mask']

        # Generate the answer from the model
        output = self.model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=conf.model_config["max_new_tokens"],
            do_sample=conf.model_config["do_sample"],
            num_beams=conf.model_config["num_beams"],
            early_stopping=conf.model_config["early_stopping"],
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode and extract the answer from the output
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded_output.split("Answer:")[1].strip() if "Answer:" in decoded_output else decoded_output

        return answer

    def batch_generate(self, queries_contexts):
        """Generate answers for a batch of queries and contexts in parallel."""
        input_texts = [f"Question: {query}\nContext: {context}\nAnswer:" for query, context in queries_contexts]

        # Encode all input texts in parallel using list comprehension
        inputs_list = [self.tokenizer(text, return_tensors="pt", padding=True) for text in input_texts]

        # Ensure each input has its own attention mask
        attention_masks = [inputs['attention_mask'] for inputs in inputs_list]

        # Generate outputs for each input in the list
        outputs = [self.model.generate(
            inputs["input_ids"],
            attention_mask=attention_mask,
            temperature=conf.model_config["temperature"],
            top_p=conf.model_config["top_p"],
            top_k=conf.model_config["top_k"],
            max_new_tokens=conf.model_config["max_new_tokens"],
            do_sample=conf.model_config["do_sample"],
            num_beams=conf.model_config["num_beams"],
            early_stopping=conf.model_config["early_stopping"],
            pad_token_id=self.tokenizer.pad_token_id  # Set the pad_token_id for generation
        ) for inputs, attention_mask in zip(inputs_list, attention_masks)]

        # Parallel decoding and answer extraction using a list comprehension
        decoded_outputs = [self.tokenizer.decode(output[0], skip_special_tokens=True) for output in outputs]

        # Batch processing using list comprehension to extract decoded and processed results
        answers = [answer.split("Answer:")[1].strip() if "Answer:" in answer else answer
                   for answer in decoded_outputs]

        return answers

    def save_model(self, base_save_path: str):
        """Save the model and tokenizer to the specified path with versioning."""
        model_dir = os.path.join(base_save_path, self.model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Determine the latest version number
        existing_versions = [d for d in os.listdir(model_dir) if d.startswith('v')]
        if existing_versions:
            latest_version = max([int(d[1:]) for d in existing_versions])
            new_version = f"v{latest_version + 1}"
        else:
            new_version = "v1"

        # Create the directory for the new version
        version_dir = os.path.join(model_dir, new_version)
        os.makedirs(version_dir, exist_ok=True)

        # Save the model and tokenizer
        self.model.save_pretrained(version_dir)
        self.tokenizer.save_pretrained(version_dir)

        print(f"Model and tokenizer saved to {version_dir}")
