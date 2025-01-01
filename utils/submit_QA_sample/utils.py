import json
from collections import Counter

def save_qa_entry(file_path, qa_data):
    """
    Save a Q&A entry to a JSONL file.
    """
    try:
        with open(file_path, 'a') as file:
            file.write(json.dumps(qa_data) + '\n')
    except Exception as e:
        raise IOError(f"Error writing to file: {e}")

def get_stats(file_path):
    """
    Read the JSONL file and calculate statistics.
    """
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        pmids = set()
        categories = Counter()

        for line in lines:
            entry = json.loads(line)
            pmids.add(entry['pmid'])
            categories[entry['category']] += 1

        return {
            "total_papers": len(pmids),
            "total_qas": len(lines),
            "categories": categories
        }
    except FileNotFoundError:
        return {
            "total_papers": 0,
            "total_qas": 0,
            "categories": {}
        }
