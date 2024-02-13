import sys
import json
from openai import OpenAI

def delete_assistant_from_api(assistant_id):
    client = OpenAI()
    response = client.beta.assistants.delete(assistant_id)
    print("API Response:", response)

def delete_assistant_from_file(file_path, assistant_id):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Filter out the assistant with the given ID
        data['data'] = [assistant for assistant in data['data'] if assistant['id'] != assistant_id]

        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
        
        print(f"Assistant with ID {assistant_id} deleted from file.")
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError:
        print("Invalid JSON file.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_assistant.py <assistant_id>")
        sys.exit(1)

    assistant_id = sys.argv[1]
    
    # Delete assistant from OpenAI API
    delete_assistant_from_api(assistant_id)
    
    # Delete assistant from local JSON file
    delete_assistant_from_file("./data/assistant_list.json", assistant_id)
