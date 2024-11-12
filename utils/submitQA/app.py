from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Path to the JSONL file
FILE_PATH = 'qa_database.jsonl'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_qa():
    # Get the form data
    question = request.form['question']
    answer = request.form['answer']
    pmid = request.form['pmid']
    doi = request.form['doi']
    category = request.form['category']

    # Create the dictionary to be saved as a JSON object
    qa_data = {
        "question": question,
        "answer": answer,
        "pmid": pmid,
        "doi": doi,
        "category": category
    }

    # Write the data to the JSONL file
    with open(FILE_PATH, 'a') as file:
        file.write(json.dumps(qa_data) + '\n')

    return jsonify({'status': 'success', 'message': 'Q&A with PMID, DOI, and Category saved successfully!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
