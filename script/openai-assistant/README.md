# README

## assistant.ipynb:
Create an instance of an OpenAI assistant and store its information (id, name, file_ids, etc.) in `./data/assistant_list.json`. To upload files in this Jupyter notebook, follow these steps:
1. Copy your files to the `./data/assistant_files` directory.
2. Run the script under the "Upload Files" section from the Jupyter notebook.

## modify_assistant.ipynb:
Modify an instance of an assistant given its id.
### Usage:
1. Modify the new parameters marked as "TODO".
2. Run the script.

The metadata of the new assistant instance is updated in `./data/assistant_list.json`.

## delete_assistant.py:
This is a Python script to delete an assistant and its instances in `./data/assistant_files` given the assistant's id.
### Usage:
```bash
python delete_assistant.py <assistant_id>
```

## Testing:
You may test the module with the following command:
```bash
python -m unittest test/test_assistant.py
```
