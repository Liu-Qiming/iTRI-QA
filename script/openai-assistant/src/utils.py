import json as js

def read_json_file(file_path):
    """
    Reads and returns the content of a JSON file.
    
    @param file_path: The path to the JSON file to be read.
    @return: A tuple containing the JSON data and None if successful, or None and an error message if an error occurred.
    """
    try:
        with open(file_path, 'r') as file:
            return js.load(file), None
    except FileNotFoundError:
        return None, f"File not found: {file_path}"
    except js.JSONDecodeError:
        return None, "Invalid JSON file."
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"

def write_json_file(file_path, data):
    """
    Writes the provided data to a JSON file.
    
    @param file_path: The path to the JSON file where data will be written.
    @param data: The data to be written into the file.
    @return: A tuple containing True and None if successful, or False and an error message if an error occurred.
    """
    try:
        with open(file_path, 'w') as file:
            js.dump(data, file, indent=4)
        return True, None
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"
