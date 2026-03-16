import pytest
from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_analyze_endpoint():
    # Uses the prompt's provided test sample
    payload = {
      "type":"email",
      "timestamp":"2026-03-16T14:22:10Z",
      "subject":"Verify your PayPal account",
      "sender":"paypal-security@mail-paypal.com",
      "content":{"text":"Your account is suspended. Click here to verify."},
      "links":[{"url":"http://paypal-login-check.com","domain":"paypal-login-check.com"}],
      "images":[],
      "attachments":[],
      "metadata":{"num_links":1,"num_images":0,"has_attachment":False}
    }
    
    response = client.post("/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check expected output keys as requested by user
    assert "classification" in data
    assert "risk_analysis" in data
    assert "recommended_actions" in data
    assert "rag_evidence" in data
    assert "metadata" in data
    
    # Inner structures validation
    assert len(data["classification"]["top_labels"]) <= 5
    assert data["classification"]["model_version"] == "facebook/bart-large-mnli"
    
    assert "risk_score" in data["risk_analysis"]
    assert "reasons" in data["risk_analysis"]
    
    assert isinstance(data["recommended_actions"], list)
    assert isinstance(data["rag_evidence"], list)
