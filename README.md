# iTRI-QA: a Toolset for Customized Question-Answer Dataset Generation Using Language Models for Enhanced Scientific Knowledge Retrieval and Preservation

**Abstract** The exponential growth of AI in science necessitates efficient and scalable solutions for retrieving and preserving research information. Here, we present the first tool of framework, Interactive Trained Research Innovator(iTRI), for the development of a customized question-answer (QA) dataset tool tailored for the needs of researchers leveraging language models (LMs). Our approach integrates curated QA datasets with a specialized research paper dataset to enhance responses' contextual relevance and accuracy. The framework comprises four key steps: (1) the generation of high-quality and human-generated QA examples, (2) the creation of a structured research paper database, (3) the fine-tuning of LLMs using domain-specific QA examples, and (4) the generation of QA dataset that align with user queries and the curated database. This pipeline provides a dynamic and domain-specific QA system that augments the utility of LLMs in academic research that will be applied for future research LLM deployment. We demonstrate the feasibility and scalability of our tool for streamlining knowledge retrieval in scientific contexts, paving the way for its integration into broader multi-disciplinary applications.

---

## Workflow

### Step 1: Fine-Tune the Model
Use the `finetune.py` script to fine-tune pre-trained LLMs on customized QA datasets.

#### Key Features of `finetune.py`:
- **Data Sampling**: Dynamically samples QA pairs of varying sizes for training.
- **LoRA Configuration**: Implements Low-Rank Adaptation (LoRA) for efficient fine-tuning with reduced computational overhead.
- **Custom Prompting**: Uses a prompt manager to ensure domain-specific QA dataset generation.
- **Fine-Tuning**: Supports multiple models (e.g., `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`) and experiments with various dataset sizes.

#### Command:
```bash
python finetune.py
```

#### Workflow Details:
1. **Data Preparation**: Samples QA pairs from the JSONL dataset and prepares them using structured prompts.
2. **LoRA Fine-Tuning**: Fine-tunes models using LoRA for computational efficiency.
3. **Evaluation**: Evaluates the fine-tuned models on a predefined validation set.
4. **Logging Results**: Logs accuracy and evaluation metrics for different QA dataset sizes and models.

#### Outputs:
- Fine-tuned models saved in `models/experiment/<model_name>_QA<size>` directories.
- Evaluation metrics logged in `models/experiment/result.txt`.

---

### Step 2: Generate QA Using Fine-Tuned Models
Use the `generate_QA.py` script to generate QA pairs from fine-tuned models.

#### Key Features of `generate_QA.py`:
- Reads abstracts from a YAML file.
- Generates structured QA pairs using the fine-tuned model.
- Outputs predictions in JSONL format.

#### Command:
```bash
python generate_QA.py --model_path <path_to_finetuned_model> --yaml_path <path_to_yaml> --output_path <output_jsonl>
```

#### Workflow Details:
1. **Model Loading**: Loads the fine-tuned LLM from the specified path.
2. **QA Generation**: Processes YAML abstracts and generates QA pairs using the model.
3. **Output Formatting**: Saves the generated Q&A in JSONL format for downstream tasks.

#### Outputs:
- JSONL file containing generated QA pairs for each abstract in the YAML file.

---

## Installation
### Prerequisites
- Python 3.8 or higher
- GPU (optional for faster training and inference)
- Libraries: `transformers`, `torch`, `datasets`, `peft`, `tqdm`

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage
### Fine-Tune Models
```bash
python finetune.py
```
This script fine-tunes the model and evaluates it on different QA dataset sizes.

### Generate QA
```bash
python generate_QA.py --model_path models/experiment/meta-llama_Llama-3.2-3B_QA25 --yaml_path input.yaml --output_path output.jsonl
```
This script generates QA pairs using a fine-tuned model.

---

## Examples
### YAML Input
```yaml
- abstract: "This study examines the role of cadmium exposure in whole-body aging..."
  doi: "10.1186/s12889-023-16643-2"
```

### JSONL Output
```json
{
  "doi": "10.1186/s12889-023-16643-2",
  "QA": {
    "question": "What is the relationship between cadmium exposure and aging biomarkers?",
    "answer": "Cadmium exposure is positively associated with phenotypic aging, mediating a proportion of 23.2%."
  },
  "category": "method"
}
```

---

## File Structure
- **`finetune.py`**: Script for fine-tuning LLMs on QA datasets.
- **`generate_QA.py`**: Script for generating QA pairs using fine-tuned models.
- **`utils/`**: Utility scripts for loading data, managing prompts, and handling file operations.
- **`models/`**: Directory for saving fine-tuned models and logging results.
- **`data/`**: Directory for input and evaluation datasets.
- **`utils/submit_QA_sample/qa_database_updated.jsonl/`**: Curated QA database by domain experts.

---

## Contributing
Contributions to enhance the functionality or performance of iTRI-QA are welcome. Please submit a pull request or open an issue for any bug reports or feature requests.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

For more details or support, please contact the repository maintainers.