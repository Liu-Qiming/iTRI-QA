# Utils Manual

## Requirements
```bash
pip install pandas xport
```

## `valid-data.py`
The `valid-data.py` script is designed to validate the input messages and prompts intended for fine-tuning the ChatGPT model. It performs various checks to ensure the data is in the correct format and provides statistics on the dataset.

### Usage

```bash
python3 utils/valid-data.py --data_path "src/training-data/msgs/msg1.jsonl"
```

### Sample Output
```text
(med-chat) qimingliu0831@QL-Linux:~/Documents/PRJ/medical-chat$ python3 utils/valid-data.py --data_path "src/training-data/msgs/msg1.jsonl"
Num examples: 3
First example:
{'role': 'system', 'content': "You are an assistant that answers professional questions as a professional Endocrinologist according to the input .XPT dataset LBXTR (Triglyceride (mg/dL)) and LBDLDL(LDL-cholesterol (mg/dL)) columns specifically. Also, refer to the scientific paper file starting with doi:. If the user is asking a question related to level conversion, perform a data analysis first to find the correlation between these two data sets and predict the user's input data with linear regression. If the user is asking a correlation question, refer to the paper."}
{'role': 'user', 'content': 'If I have an LDL level of 135 mg/dL, what should be my expected Triglyceride level'}
{'role': 'assistant', 'content': 'It is highly likely that your high LDL level will come with a high Triglyceride level with ${YOUR_PREDICTED_VALUE_OF_TRIGLYCERIDE_LEVEL}'}
No errors found
Num examples missing system message: 0
Num examples missing user message: 0

#### Distribution of num_messages_per_example:
min / max: 3, 3
mean / median: 3.0, 3.0
p5 / p95: 3.0, 3.0

#### Distribution of num_total_tokens_per_example:
min / max: 170, 190
mean / median: 182.0, 186.0
p5 / p95: 173.2, 189.2

#### Distribution of num_assistant_tokens_per_example:
min / max: 20, 35
mean / median: 28.666666666666668, 31.0
p5 / p95: 22.2, 34.2

0 examples may be over the 4096 token limit, they will be truncated during fine-tuning
Dataset has ~546 tokens that will be charged for during training
By default, you'll train for 25 epochs on this dataset
By default, you'll be charged for ~13650 tokens

```

## `xpt_to_csv_convertor.py`
The `xpt_to_csv_convertor.py` script is designed to convert .XPT into .csv.

### Usage

```bash
python3 xport_to_csv_converter.py --file_name "YourXportFile.XPT" --rows 50
```

