from flask import Flask, render_template, request, jsonify
import json
from collections import Counter

app = Flask(__name__)

# Path to the JSONL file
FILE_PATH = 'qa_database.jsonl'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_qa():
    question = request.form['question']
    answer = request.form['answer']
    pmid = request.form['pmid']
    doi = request.form['doi']
    category = request.form['category']

    qa_data = {
        "question": question,
        "answer": answer,
        "pmid": pmid,
        "doi": doi,
        "category": category
    }

    # Write the data to JSONL file
    with open(FILE_PATH, 'a') as file:
        file.write(json.dumps(qa_data) + '\n')

    return jsonify({'status': 'success', 'message': 'Q&A saved successfully!'})

@app.route('/stats', methods=['GET'])
def get_stats():
    try:
        with open(FILE_PATH, 'r') as file:
            lines = file.readlines()

        pmids = set()
        categories = Counter()

        for line in lines:
            entry = json.loads(line)
            pmids.add(entry['pmid'])  # Collect unique PMIDs
            categories[entry['category']] += 1  # Count categories

        stats = {
            "total_papers": len(pmids),
            "total_qas": len(lines),
            "categories": categories
        }
        return jsonify(stats)
    except FileNotFoundError:
        return jsonify({
            "total_papers": 0,
            "total_qas": 0,
            "categories": {}
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
