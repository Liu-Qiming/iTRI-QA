import os

from src.utils import *

def update_assistant_list(list_file, my_assistant):
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
    list_file, error = read_json_file(file_path)
    if error:
        return None, error

    updated_list = update_assistant_list(list_file, my_assistant)

    success, write_error = write_json_file(file_path, updated_list)
    if not success:
        return None, write_error

    return updated_list, None


def list_assistant_file_names(assistant_file_path):
    file_names = [name for name in os.listdir(assistant_file_path) if os.path.isfile(os.path.join(assistant_file_path, name))]
    return file_names


# Function to retrieve all files for "assistant"
def get_assitant_files(json_data):
    retrieved_files = []
    for item in json_data.get('data', []):
        if item.get('purpose') == "assistants":
            retrieved_files.append(item.get("id"))
    
    return retrieved_files

# Function to create and dump messages into a JSON file
def create_and_dump_messages(messages, file_path):
    existing_data = []

    # Read existing data if the file exists
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as file:
                existing_data = js.load(file)
        except js.JSONDecodeError:
            print("Invalid JSON file. Starting fresh.")
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")

    # Update the data with new messages
    if messages:
        existing_data.extend(messages)

    # Write the updated data back to the file
    success, error = write_json_file(file_path, existing_data)
    if not success:
        print(error)
    else:
        print(f"Updated file at {file_path} with {len(messages)} messages")


def find_assistant_by_name(file_path, target_assistant_name):
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

