from pybtex.database import parse_string
import pandas as pd
import json

from pybtex.database import parse_string
import pandas as pd

def read_bib_file(bib_file):
    """Read a BibTeX file and return a pandas DataFrame."""
    with open(bib_file, 'r', encoding='utf-8') as bibtex_file:
        bibtex_str = bibtex_file.read()

    try:
        bib_data = parse_string(bibtex_str, 'bibtex')  # Parse the BibTeX string
    except Exception as e:
        raise ValueError(f"Error parsing the BibTeX file: {e}")

    # Extract BibTeX data into a list of dictionaries
    bib_entries = []
    for key, entry in bib_data.entries.items():
        fields = dict(entry.fields)  # Convert OrderedCaseInsensitiveDict to dict
        fields['key'] = key  # Add the BibTeX entry key
        fields['authors'] = " and ".join(
            str(person) for person in entry.persons.get("author", [])
        )  # Combine author names if available
        bib_entries.append(fields)

    return pd.DataFrame(bib_entries)

def read_json_file(json_file):
    """Read a JSON file and return a pandas DataFrame."""
    with open(json_file, 'r', encoding='utf-8') as json_file:
        try:
            json_data = json.load(json_file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing the JSON file: {e}")

    # Convert JSON data into a DataFrame
    json_entries = [
        {"key": key, **value} for key, value in json_data.items()
    ]  # Assuming JSON is a dictionary
    return pd.DataFrame(json_entries)

def merge_dataframes(bib_df, json_df):
    """Merge BibTeX and JSON DataFrames on the 'key' column."""
    merged_df = pd.merge(bib_df, json_df, on='key', how='left')
    return merged_df

# Example usage
if __name__ == "__main__":
    bibtex_file_path = "/Users/maotian/Documents/Project/Project-GPT/iTRI-GPT/utils/load_abstract_db/pubmed-telomerenh-set.bib"
    json_file_path = "path/to/your/file.json"

    try:
        # Read individual files
        bib_df = read_bib_file(bibtex_file_path)
        json_df = read_json_file(json_file_path)

        # Merge the DataFrames
        merged_df = merge_dataframes(bib_df, json_df)

        # Display the merged DataFrame
        print("Merged DataFrame:")
        print(merged_df.head())

        # Optionally, save the merged DataFrame to a file
        merged_df.to_csv("merged_data.csv", index=False)

    except Exception as e:
        print(f"An error occurred: {e}")
