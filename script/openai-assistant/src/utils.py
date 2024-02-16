import json as js

def read_json_file(file_path):
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
    try:
        with open(file_path, 'w') as file:
            js.dump(data, file, indent=4)
        return True, None
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"