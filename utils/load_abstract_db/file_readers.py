import os
from pybtex.database import parse_string
import pandas as pd
import json

def read_bib_file(bib_file):
    """
    Read a BibTeX file and return a pandas DataFrame.

    Parameters:
        bib_file (str): Path to the BibTeX file.

    Returns:
        pd.DataFrame: DataFrame containing BibTeX entries.
    """
    if not os.path.exists(bib_file):
        raise FileNotFoundError(f"The file {bib_file} does not exist.")

    with open(bib_file, 'r', encoding='utf-8') as bibtex_file:
        bibtex_str = bibtex_file.read()

    try:
        bib_data = parse_string(bibtex_str, 'bibtex')
    except Exception as e:
        raise ValueError(f"Error parsing the BibTeX file: {e}")

    bib_entries = []
    for key, entry in bib_data.entries.items():
        fields = dict(entry.fields)
        fields['key'] = key
        fields['authors'] = " and ".join(
            str(person) for person in entry.persons.get("author", [])
        )
        bib_entries.append(fields)

    return pd.DataFrame(bib_entries)

def read_json_file(json_file):
    """
    Read a JSON file and return a pandas DataFrame.

    Parameters:
        json_file (str): Path to the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing JSON data.
    """
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"The file {json_file} does not exist.")

    with open(json_file, 'r', encoding='utf-8') as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing the JSON file: {e}")

    if isinstance(json_data, list):
        return pd.DataFrame(json_data)
    elif isinstance(json_data, dict):
        return pd.DataFrame([{"key": key, **value} for key, value in json_data.items()])
    else:
        raise ValueError("Unsupported JSON format. Expected a list or dictionary.")
