import os
import json
import pandas as pd


def read_csv_file(file_path, num_rows=5):
    """
    Reads the first `num_rows` rows from a CSV file and returns a pandas DataFrame.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    - num_rows (int): Number of rows to read. Default is 5.
    
    Returns:
    - pd.DataFrame: DataFrame containing the first `num_rows` rows of the CSV.
    """
    try:
        df = pd.read_csv(file_path, nrows=num_rows)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return df

def format_dataframe(df):
    """
    Formats a DataFrame by prepending column names to their corresponding values.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to be formatted.
    
    Returns:
    - str: The formatted DataFrame as a string.
    """
    formatted_rows = []
    for _, row in df.iterrows():
        formatted_row = "\n".join([f"{col}: {row[col]}" for col in df.columns])
        formatted_rows.append(formatted_row)
    # Join each formatted row with a new line
    formatted_output = "\n\n".join(formatted_rows)
    return formatted_output

def process_csv_data(file_path, num_rows=5):
    """
    Reads and formats the first `num_rows` rows from the CSV file.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    - num_rows (int): Number of rows to read and format. Default is 5.
    
    Returns:
    - str: A string representing the formatted CSV data.
    """
    df = read_csv_file(file_path, num_rows)
    formatted_output = format_dataframe(df)
    return formatted_output

def load_json_data(file_path):
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data