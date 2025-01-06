from transformers import pipeline

from QA_augmentation import QADataAugmentor


class ParaphrasingAugmentor(QADataAugmentor):
    """
    Augmentor for paraphrasing questions and answers.
    """
    def __init__(self):
        super().__init__()
        self.paraphrase_pipeline = pipeline(
            "text2text-generation",
            model="Vamsi/T5_Paraphrase_Paws",  # Use a model fine-tuned for paraphrasing
            device=0 if self.device == "cuda" else -1
        )

    def augment(self, entry):
        """
        Apply paraphrasing to a single entry.

        Parameters:
        - entry (dict): A dictionary containing 'question', 'answer', 'pmid', and 'abstract'.

        Returns:
        - dict: Paraphrased version of the input entry.
        """
        # Explicitly define the paraphrasing task in the input prompt
        paraphrase_prompt = lambda text: f"paraphrase: {text}"

        # Paraphrase the question
        paraphrased_question = self.paraphrase_pipeline(
            paraphrase_prompt(entry["question"]),
            max_length=128,
            num_return_sequences=1,
            truncation=True
        )[0]["generated_text"]

        # Paraphrase the answer
        paraphrased_answer = self.paraphrase_pipeline(
            paraphrase_prompt(entry["answer"]),
            max_length=128,
            num_return_sequences=1,
            truncation=True
        )[0]["generated_text"]

        return {
            "question": paraphrased_question,
            "answer": paraphrased_answer,
            "pmid": entry["pmid"],
            "abstract": entry["abstract"]
        }
