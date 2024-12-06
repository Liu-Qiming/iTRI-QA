import random

from QA_augmentation import QADataAugmentor


class SynonymReplacementAugmentor(QADataAugmentor):
    """
    Augmentor for replacing words with synonyms in questions and answers.
    """
    def __init__(self):
        super().__init__()
        self.synonym_dict = {
            "important": ["crucial", "vital"],
            "study": ["research", "analysis"],
            "relationship": ["connection", "association"]
        }

    def replace_synonyms(self, text):
        words = text.split()
        augmented_words = [
            random.choice(self.synonym_dict[word]) if word in self.synonym_dict else word
            for word in words
        ]
        return " ".join(augmented_words)

    def augment(self, entry):
        """
        Apply synonym replacement to a single entry.

        Parameters:
        - entry (dict): A dictionary containing 'question', 'answer', 'pmid', and 'abstract'.

        Returns:
        - dict: Synonym-replaced version of the input entry.
        """
        augmented_question = self.replace_synonyms(entry["question"])
        augmented_answer = self.replace_synonyms(entry["answer"])

        return {
            "question": augmented_question,
            "answer": augmented_answer,
            "pmid": entry["pmid"],
            "abstract": entry["abstract"]
        }
