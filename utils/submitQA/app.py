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

    # Create the dictionary to be saved as a JSON object
    qa_data = {
        "question": question,
        "answer": answer,
        "pmid": pmid
    }

    # Write the data to the JSONL file
    with open(FILE_PATH, 'a') as file:
        file.write(json.dumps(qa_data) + '\n')

    return jsonify({'status': 'success', 'message': 'Q&A with PMID saved successfully!'})

if __name__ == '__main__':
    app.run(debug=True)
