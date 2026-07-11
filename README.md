---
title: Email Security AI
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ Email Security AI

AI-powered email security service that detects phishing, scores risk, and recommends safe actions — combining zero-shot classification, a hybrid ML + rules risk engine, and Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-async-00a393)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

**Email Security AI** is a REST API microservice that takes a structured email as input and returns a full security assessment: what kind of threat it is, a 0–100 risk score with human-readable reasons, supporting evidence from a knowledge base, and concrete recommended actions. It's built as a modular FastAPI backend with Pydantic-validated contracts, containerized for Hugging Face Spaces.

---

## ✨ Key Features

- **🚀 FastAPI backend** — async REST API with automatic request/response validation and interactive OpenAPI docs at `/docs`.
- **🧠 Four-stage inference pipeline:**
  1. **Classification** — zero-shot labeling of the email (phishing / scam / fraud / malware / spam / benign).
  2. **Risk analysis** — a hybrid engine combining a zero-shot NLI model with rule-based heuristics into a 0–100 score.
  3. **RAG retrieval** — semantic search over a cybersecurity knowledge base for supporting evidence.
  4. **Action generation** — an instruction-tuned model produces grounded, actionable safety steps.
- **📚 RAG pipeline** — embeds text with `sentence-transformers/all-MiniLM-L6-v2` and retrieves from a local **ChromaDB** vector store.
- **🔍 Explainable by design** — every point of risk comes with a reason; advice is grounded in retrieved evidence.
- **🐳 Dockerized** — one container, runs anywhere, deploys to Hugging Face Spaces on port `7860`.

---

## 🏗️ Architecture

```
POST /analyze (JSON)
      │
      ▼
Pydantic validation ──► Security Pipeline
                              │
     ┌────────────┬───────────┼────────────┬─────────────┐
     ▼            ▼           ▼             ▼
  Model 1      Model 2       RAG         Model 3
  classify  →  risk score →  evidence  →  actions
     │            │           │             │
     └────────────┴───────────┴─────────────┘
                              │
                              ▼
                    Structured AnalyzeResponse (JSON)
```

| Stage | Component | Model / Technique |
|-------|-----------|-------------------|
| 1 | Classifier | `facebook/bart-large-mnli` (zero-shot) |
| 2 | Risk engine | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` + rule heuristics |
| 3 | RAG | ChromaDB + `sentence-transformers/all-MiniLM-L6-v2` |
| 4 | Action generator | `google/flan-t5-base` (text-to-text) |

**Flow:** a client POSTs an `AnalyzeRequest` → Pydantic validates it → the pipeline runs Model 1 → Model 2 → RAG → Model 3 → the API returns a structured `AnalyzeResponse` with classification, risk analysis, recommended actions, RAG evidence, and run metadata.

---

## 💻 API Usage

### Endpoint

`POST /analyze`

### Request

```bash
curl -X POST 'http://localhost:7860/analyze' \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "email",
    "subject": "Verify your PayPal account",
    "sender": {
      "email": "paypal-security@mail-paypal.com",
      "domain": "paypal.com"
    },
    "content": {
      "text": "Your account is suspended. Click here to verify your identity."
    },
    "links": [
      { "url": "http://paypal-login-check.com", "domain": "paypal-login-check.com" }
    ],
    "metadata": { "has_attachment": false }
  }'
```

### Response

```json
{
  "classification": {
    "top_labels": [
      { "label": "phishing", "probability": 0.82 },
      { "label": "scam", "probability": 0.09 }
    ],
    "model_version": "model1-v1"
  },
  "risk_analysis": {
    "risk_score": 78,
    "risk_level": "high",
    "reasons": [
      "Message classified as potential phishing",
      "Suspicious verification-style domain detected",
      "Sender domain mismatch detected"
    ],
    "model_version": "model2-v3"
  },
  "recommended_actions": [
    "Do not click any links in the message",
    "Verify the sender through the official website",
    "Report the message as phishing"
  ],
  "rag_evidence": [
    "Phishing emails try to trick recipients into revealing credentials..."
  ],
  "metadata": {
    "processing_time_ms": 940,
    "models": {
      "model1": "facebook/bart-large-mnli",
      "model2": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
      "model3": "google/flan-t5-base"
    }
  }
}
```

**Risk levels:** `safe` (score < 25) · `suspicious` (25–49) · `risky` (50–74) · `high` (≥ 75).

Other endpoints: `GET /` and `GET /health` (health checks), `GET /docs` (interactive Swagger UI).

---

## 🛠️ Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the API (from the repo root)
uvicorn app.app:app --reload --port 7860

# 3. Open the interactive docs
open http://localhost:7860/docs
```

> On first run the app downloads the HuggingFace models, which can take a few minutes. Watch the logs for **"Model singletons initialized and ready"** before sending requests.

### Run with Docker

```bash
docker build -t email-security-ai .
docker run -p 7860:7860 email-security-ai
curl http://localhost:7860/health   # -> {"status":"ok","ready":true}
```

### Run the tests

```bash
pytest
```

---

## 📁 Project Structure

```
app/
├── app.py                 # FastAPI app: CORS, startup warm-up, routes
├── routes/analyze.py      # POST /analyze
├── pipelines/             # orchestrates the 4-stage pipeline
├── models/                # model1 (classify), model2 (risk), model3 (actions)
├── rag/                   # Chroma vector store + knowledge loader
├── embeddings/            # MiniLM embedding wrapper
├── schemas/               # Pydantic request/response contracts
├── config/                # centralized paths + model names
├── utils/                 # text cleaning
└── data/knowledge_base/   # cybersecurity guidance indexed by RAG
```

---

## ⚙️ Tech Stack

FastAPI · Pydantic v2 · LangChain · HuggingFace Transformers · sentence-transformers · ChromaDB · PyTorch · Docker
