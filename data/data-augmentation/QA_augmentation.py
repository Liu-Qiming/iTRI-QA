import torch
from abc import ABC, abstractmethod


class QADataAugmentor(ABC):
    def __init__(self):
        """
        Initialize the augmentor with a dataset and hardware device selection.

        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def augment(self):
        """
        Abstract method to apply augmentation.

        Returns:
        - Augmented dataset as a list of dictionaries.
        """
        raise NotImplementedError("Subclasses must implement the 'augment' method.")
