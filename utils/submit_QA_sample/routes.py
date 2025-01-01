from flask import Flask, render_template, request, jsonify
from utils import save_qa_entry, get_stats
from config import JSONL_FILE_PATH

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_qa():
    question = request.form['question'].strip()
    answer = request.form['answer'].strip()
    pmid = request.form['pmid'].strip()
    doi = request.form['doi'].strip()
    category = request.form['category'].strip()

    if not question or not answer or not pmid or not doi:
        return jsonify({'status': 'error', 'message': 'All fields must be filled.'}), 400

    qa_data = {
        "question": question,
        "answer": answer,
        "pmid": pmid,
        "doi": doi,
        "category": category
    }

    try:
        save_qa_entry(JSONL_FILE_PATH, qa_data)
        return jsonify({'status': 'success', 'message': 'Q&A saved successfully!'})
    except IOError as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    stats = get_stats(JSONL_FILE_PATH)
    return jsonify(stats)
