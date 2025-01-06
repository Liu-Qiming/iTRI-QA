import json
import yaml

# Paths to input files
jsonl_file_path = "/Users/maotian/Documents/Project/Project-GPT/iTRI-GPT/utils/submit_qa_sample/qa_database.jsonl"
yaml_file_path = "/Users/maotian/Documents/Project/Project-GPT/iTRI-GPT/utils/load_abstract_db/output.yaml"
output_file_path = "/Users/maotian/Documents/Project/Project-GPT/iTRI-GPT/utils/submit_qa_sample/qa_database_updated.jsonl"

# Load YAML file
def load_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        return yaml.safe_load(file)

# Load JSONL file
def load_jsonl(jsonl_file_path):
    data = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any leading/trailing whitespace
            if not line:  # Skip empty lines
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line}\nError: {e}")
    return data

# Write JSONL file
def write_jsonl(data, output_file_path):
    with open(output_file_path, 'w') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

# Update JSONL with abstracts
def update_jsonl_with_abstracts(jsonl_data, yaml_data):
    doi_to_abstract = {entry['doi']: entry.get('abstract', '') for entry in yaml_data}
    for jsonl_entry in jsonl_data:
        jsonl_entry['abstract'] = doi_to_abstract.get(jsonl_entry['doi'], "")
    return jsonl_data

# Main process
def main():
    yaml_data = load_yaml(yaml_file_path)
    jsonl_data = load_jsonl(jsonl_file_path)

    updated_data = update_jsonl_with_abstracts(jsonl_data, yaml_data)

    write_jsonl(updated_data, output_file_path)
    print(f"Updated JSONL file saved to {output_file_path}")

if __name__ == "__main__":
    main()
