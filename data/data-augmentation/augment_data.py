import json
from tqdm import tqdm

from paraphrasing import ParaphrasingAugmentor
from synonym_replace import SynonymReplacementAugmentor
from backtranslation import BacktranslationAugmentor


class DataAugmentor:
    def __init__(self, input_path, output_path, n_samples):
        """
        Initialize the DataAugmentor.

        Parameters:
        - input_path: Path to the input JSONL file.
        - output_path: Path to the output JSONL file.
        - n_samples: Number of augmentation repetitions for each data entry.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.n_samples = n_samples

    def load_data(self):
        """
        Load the input JSONL data file.
        """
        with open(self.input_path, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def save_data(self, augmented_data):
        """
        Save augmented data to the output JSONL file.
        """
        with open(self.output_path, "w") as f:
            for entry in augmented_data:
                f.write(json.dumps(entry) + "\n")

    def run_augmentation(self):
        """
        Run all augmentation methods and save the results.
        """
        data = self.load_data()
        print(f"Loaded {len(data)} records from {self.input_path}.")

        # Initialize augmentors
        print("Initializing augmentation methods...")
        paraphraser = ParaphrasingAugmentor()
        synonym_replacer = SynonymReplacementAugmentor()
        backtranslator = BacktranslationAugmentor()

        # Collect all augmented data
        all_augmented_data = []
        for _ in tqdm(range(self.n_samples)):
            for entry in data:
                # Paraphrasing Augmentation
                paraphrased_entry = paraphraser.augment(entry)
                all_augmented_data.append(paraphrased_entry)

                # Synonym Replacement Augmentation
                synonym_replaced_entry = synonym_replacer.augment(entry)
                all_augmented_data.append(synonym_replaced_entry)

                # Backtranslation Augmentation
                backtranslated_entry = backtranslator.augment(entry)
                all_augmented_data.append(backtranslated_entry)

        print(f"Generated {len(all_augmented_data)} augmented examples.")

        # Save to output
        self.save_data(all_augmented_data)
        print(f"Augmented data saved to {self.output_path}.")


if __name__ == "__main__":
    input_path = "data/QA_data/qa_abstract.jsonl"
    output_path = "data/QA_data/augmented/augmented_qa_abstract.jsonl"
    n_samples = 1000

    augmentor = DataAugmentor(input_path, output_path, n_samples)
    augmentor.run_augmentation()
