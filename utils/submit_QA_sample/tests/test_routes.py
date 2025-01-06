import pytest
from routes import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200

def test_submit_qa(client):
    data = {
        "question": "Test Question",
        "answer": "Test Answer",
        "pmid": "123456",
        "doi": "10.1234/example",
        "category": "knowledge"
    }
    response = client.post('/submit', data=data)
    assert response.status_code == 200
    assert response.json['status'] == 'success'

def test_stats(client):
    response = client.get('/stats')
    assert response.status_code == 200
    assert 'total_papers' in response.json
