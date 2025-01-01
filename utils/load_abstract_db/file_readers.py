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


def read_output_yaml_file(yaml_file):
    """
    Read a YAML file and return abstracts and DOIs.

    Parameters:
        yaml_file (str): Path to the YAML file.

    Returns:
        list: A list of dictionaries containing abstracts and DOIs.
    """
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    results = []
    for entry in data:
        abstract = entry.get('abstract', None)
        doi = entry.get('doi', None)
        if abstract or doi:
            results.append({'abstract': abstract, 'doi': doi})

    return results
