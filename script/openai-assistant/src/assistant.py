import os
import json as js

from src.utils import read_json_file, write_json_file

def update_assistant_list(list_file, my_assistant):
    """
    Updates the assistant list with the provided assistant data.
    
    @param list_file: The current list of assistants.
    @param my_assistant: The assistant data to add or update in the list.
    @return: The updated list of assistants.
    """
    my_assistant_id = my_assistant.get('id')

    if not list_file:
        list_file = {"data": []}

    for item in list_file['data']:
        if item.get('id') == my_assistant_id:
            item.update(my_assistant)
            break
    else:
        list_file['data'].append(my_assistant)

    return list_file

def manage_assistant_file(file_path, my_assistant):
    """
    Manages the assistant data file by updating it with the provided assistant data.
    
    @param file_path: The path to the assistant data file.
    @param my_assistant: The assistant data to add or update in the file.
    @return: The updated assistant list and None if successful, or None and an error message if failed.
    """
    list_file, error = read_json_file(file_path)
    if error:
        return None, error

    updated_list = update_assistant_list(list_file, my_assistant)

    success, write_error = write_json_file(file_path, updated_list)
    if not success:
        return None, write_error

    return updated_list, None

def list_assistant_file_names(assistant_file_path):
    """
    Lists the names of files in the specified directory.
    
    @param assistant_file_path: The path to the directory containing assistant files.
    @return: A list of filenames found in the specified directory.
    """
    file_names = [name for name in os.listdir(assistant_file_path) if os.path.isfile(os.path.join(assistant_file_path, name))]
    return file_names

def get_assitant_files(json_data):
    """
    Retrieves a list of file IDs marked for "assistant" purpose from the given JSON data.
    
    @param json_data: The JSON data containing files information.
    @return: A list of file IDs for "assistant".
    """
    retrieved_files = []
    for item in json_data.get('data', []):
        if item.get('purpose') == "assistants":
            retrieved_files.append(item.get("id"))
    
    return retrieved_files

def create_and_dump_messages(messages, file_path):
    """
    Creates and dumps messages into a JSON file, updating the existing data.
    
    @param messages: The list of messages to be added to the file.
    @param file_path: The path to the JSON file to be updated.
    """
    existing_data = []

    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as file:
                existing_data = js.load(file)
        except js.JSONDecodeError:
            print("Invalid JSON file. Starting fresh.")
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")

    if messages:
        existing_data.extend(messages)

    success, error = write_json_file(file_path, existing_data)
    if not success:
        print(error)
    else:
        print(f"Updated file at {file_path} with {len(messages)} messages")

def find_assistant_by_name(file_path, target_assistant_name):
    """
    Finds an assistant by name from the assistant list in the specified file.
    
    @param file_path: The path to the assistant data file.
    @param target_assistant_name: The name of the assistant to find.
    @return: The ID of the assistant if found, None otherwise.
    """
    list_file, error = read_json_file(file_path)
    if error:
        print(error)
        return None

    if not list_file or 'data' not in list_file or not list_file['data']:
        print("No Assistant Exists")
        return None

    for item in list_file.get('data', []):
        if item.get('name') == target_assistant_name:
            assistant_id = item.get("id")
            print("Assistant name matched:")
            print(f"Assistant ID: {assistant_id}")
            print(f"Target Assistant Name: {target_assistant_name}")
            return assistant_id

    print(f"No assistant found with the name: {target_assistant_name}")
    return None
