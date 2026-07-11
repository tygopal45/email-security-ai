"""
API smoke tests.

These exercise the whole stack end-to-end through FastAPI's TestClient: they hit
the real endpoints and assert the response *shape* (the right keys and types).
They're integration-style checks that the pipeline wires together correctly —
not a measure of model accuracy.
"""
from fastapi.testclient import TestClient

from app.app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_endpoint():
    payload = {
        "type": "email",
        "timestamp": "2026-03-16T14:22:10Z",
        "subject": "Verify your PayPal account",
        "sender": {
            "name": "PayPal Security",
            "email": "paypal-security@mail-paypal.com",
            "domain": "paypal.com",
        },
        "content": {"text": "Your account is suspended. Click here to verify."},
        "links": [
            {"url": "http://paypal-login-check.com", "domain": "paypal-login-check.com"}
        ],
        "images": [],
        "attachments": [],
        "metadata": {"num_links": 1, "num_images": 0, "has_attachment": False},
    }

    response = client.post("/analyze", json=payload)

    assert response.status_code == 200
    data = response.json()

    # Top-level response keys
    assert "classification" in data
    assert "risk_analysis" in data
    assert "recommended_actions" in data
    assert "rag_evidence" in data
    assert "metadata" in data

    # Inner structures
    assert len(data["classification"]["top_labels"]) <= 5
    assert data["classification"]["model_version"] == "model1-v1"

    assert "risk_score" in data["risk_analysis"]
    assert "reasons" in data["risk_analysis"]

    assert isinstance(data["recommended_actions"], list)
    assert isinstance(data["rag_evidence"], list)
