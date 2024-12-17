from transformers import MarianMTModel, MarianTokenizer
import torch
from QA_augmentation import QADataAugmentor


class BacktranslationAugmentor(QADataAugmentor):
    def __init__(self):
        super().__init__()
        self.device = self.device
        self.translator_en_to_fr = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-fr"
        ).to(self.device)
        self.translator_fr_to_en = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-fr-en"
        ).to(self.device)
        self.tokenizer_en_to_fr = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-fr"
        )
        self.tokenizer_fr_to_en = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-fr-en"
        )

    def translate(self, text, model, tokenizer):
        """
        Translate text using the specified model and tokenizer.

        Parameters:
        - text: String input to translate.
        - model: MarianMTModel for translation.
        - tokenizer: MarianTokenizer for the model.

        Returns:
        - Translated text.
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        outputs = model.generate(**inputs)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def augment(self, data_entry):
        """
        Apply backtranslation to a single data entry.

        Parameters:
        - data_entry: Dictionary containing 'question', 'answer', etc.

        Returns:
        - Augmented data entry with backtranslated fields.
        """
        try:
            # Step 1: Translate question and answer to French
            question_to_french = self.translate(
                data_entry["question"], self.translator_en_to_fr, self.tokenizer_en_to_fr
            )
            answer_to_french = self.translate(
                data_entry["answer"], self.translator_en_to_fr, self.tokenizer_en_to_fr
            )

            # Step 2: Translate question and answer back to English
            question_back_to_en = self.translate(
                question_to_french, self.translator_fr_to_en, self.tokenizer_fr_to_en
            )
            answer_back_to_en = self.translate(
                answer_to_french, self.translator_fr_to_en, self.tokenizer_fr_to_en
            )
            # Return the backtranslated entry
            return {
                "question": question_back_to_en,
                "answer": answer_back_to_en,
                "pmid": data_entry["pmid"],
                "abstract": data_entry["abstract"],
            }
        except Exception as e:
            print(f"Error in backtranslation: {e}")
            return data_entry  # Return the original entry if backtranslation fails
