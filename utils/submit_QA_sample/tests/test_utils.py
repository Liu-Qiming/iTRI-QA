import pytest
from utils import save_qa_entry, get_stats
import os

TEST_FILE = 'test_qa_database.jsonl'

@pytest.fixture
def setup_test_file():
    # Set up a test file
    with open(TEST_FILE, 'w') as file:
        file.write('{"question": "Test Q1", "answer": "Test A1", "pmid": "123", "doi": "10.1234", "category": "knowledge"}\n')
    yield
    # Tear down test file
    os.remove(TEST_FILE)

def test_save_qa_entry(setup_test_file):
    qa_data = {
        "question": "Test Q2",
        "answer": "Test A2",
        "pmid": "456",
        "doi": "10.5678",
        "category": "method"
    }
    save_qa_entry(TEST_FILE, qa_data)

    with open(TEST_FILE, 'r') as file:
        lines = file.readlines()

    assert len(lines) == 2  # Should have 2 entries

def test_get_stats(setup_test_file):
    stats = get_stats(TEST_FILE)
    assert stats['total_papers'] == 1
    assert stats['total_qas'] == 1
    assert stats['categories']['knowledge'] == 1
