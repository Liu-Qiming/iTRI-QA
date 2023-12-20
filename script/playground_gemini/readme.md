# Gemini Playground

This project contains a Jupyter notebook designed for experimenting with the Gemini Pro API, focusing on Knowledge Retrieval (KR). The notebook is intended for use in Google Colab and involves steps for setup and execution.
Currently, it is free for knowledge retrieval and any query. 

## Getting Started

Follow these steps to set up and run the demo:

1. **Download the Notebook:**
   - Download `gemini_KR.ipynb` to your local machine.

2. **Get the Data File:**
   - Download `2023-12-15_abstract-38041119-set.txt` from the shared drive.

3. **Create Folder Structure on your Very First Web Page on Google Drive:**

```
Colab
│
├── src
│   └── 2023-12-15_abstract-38041119-set.txt
│
└── gemini_KR.ipynb
```


4. **Acquire Gemini API Key:**
- Apply for a Gemini API key at [Gemini API](https://makersuite.google.com/).
- Ensure to keep a local copy of the API key.

5. **Setup API Key in Colab:**
- In Google Colab, click on the "secret" icon (key icon) on the left side.
- Copy and paste your API key into the value field and name it `GOOGLE_API_KEY`.
- Toggle the "Notebook access" option to 'Yes'.

6. **Modify the Query:**
- Change the query in the notebook to your desired query.

7. **Run the Notebook:**
- Execute all cells in the notebook.

## Prerequisites

- A Google account for accessing Google Colab.

## Additional Information

- Keep your API keys secure and do not share them publicly.

## Advanced

Some advanced instructions for this notebook:

### split_document(file_path, chunk_size_bytes=2000)
**Functionality**

- The function reads a text file from a given file path.
- It then splits the file's content into chunks, each with a specified maximum size in bytes.
- Each chunk is returned as a separate document segment, with its own title and content.

**Usage**

```python
file_path = "your/file/path/to/2023-12-15_abstract-38041119-set.txt/.txt"
split_docs = split_document(file_path, chunk_size_bytes=2000)
```

**Note**
- The chunk size cannot exceed 8000 because of the LLM input limit.
- After some research, 2000 seems like a good chunk to feed in because Gemini doesn't seem like to be good at handle a larger chunk.
- The chunk size still needs further investigation.

### find_best_passage(query, dataframe)

Finds the most relevant passage from a dataframe for a given query by computing 
  the distances between the query and each document in the dataframe using the dot product
  with highest the most relevant.

**Functionality**

- The function takes a `query` and a `dataframe` as inputs.
- It computes the semantic embeddings for the query.
- Then, it calculates the dot product between the query embedding and the embeddings of each document in the dataframe.
- The function identifies the document with the highest dot product value, indicating the closest semantic relevance to the query.
- It returns the text of the most relevant document.

**Usage**

```python
query = "Your search query"
relevant_text = find_best_passage(query, dataframe)

print("Most relevant passage:", relevant_text)
```

**Note**

This function is from the official doc and is tuned and pretty accurate. Might need some further research on how good it performs on multiple papers on a same topic.
