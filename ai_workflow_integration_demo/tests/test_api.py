from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_semantic_search():
    r = client.post('/semantic-search', json={'query': 'ACH transfer', 'k': 2})
    assert r.status_code == 200
    assert len(r.json()) <= 2

def test_classify():
    payload = {'text': 'Please buy gift cards for me', 'threshold': 0.5}
    r = client.post('/classify-fraud', json=payload)
    assert r.status_code == 200
    assert 'label' in r.json()

def test_ask_llm():
    r = client.post('/ask-llm', json={'query': 'How do ACH transfers work?'})
    assert r.status_code == 200
    assert 'answer' in r.json()
