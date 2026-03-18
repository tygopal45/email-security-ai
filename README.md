---
title: Email Security AI
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# Email Security AI
# 🛡️ Email Security AI
AI-powered email security system that detects phishing emails, analyzes risk, and suggests safe actions using machine learning and RAG.
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-00a393)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
**Email Security AI** is an intelligent, high-performance REST API microservice designed to detect phishing emails, assess threat risk levels, and dynamically generate safety recommendations. 
Leveraging a multi-model approach and Retrieval-Augmented Generation (RAG), this backend engine processes JSON payloads representing email metadata and returns deterministic, structured analysis for downstream or frontend consumption.
---
## ✨ Key Features
- **🚀 FastAPI Backend**: Highly scalable, async REST API processing structured JSON email payloads.
- **🧠 Multi-Model Orchestration**: Utilizes a 3-stage inference pipeline:
  1. **Phishing Classification**: Categorizes email intent.
  2. **Risk Assessment**: Scores emails from 1-100 indicating risk severity.
  3. **Action Recommendation**: Predicts the best preventative action for the user.
- **📚 RAG Pipeline integration**: Embeds email content using `sentence-transformers` and searches local `ChromaDB` vector storage to provide contextual historic threat evidence.
- **🐳 Dockerized Deployment**: Fully containerized and optimized for Hugging Face Spaces.
---
## 🏗️ Architecture
1. **Client** POSTs a JSON `AnalyzeRequest` to `/analyze`.
2. **Pydantic Validation** ensures the incoming payload is well-formed.
3. The **Security Pipeline** triggers, executing:
   - `model1_classifier`: Classifies the text.
   - `model2_risk`: Assesses risk based on content and classifications.
   - `rag_engine`: Retrieves evidence from ChromaDB using semantic search.
   - `model3_action`: Synthesizes final actions.
4. **API** returns a heavily structured `AnalyzeResponse` containing classification, risk score, recommended actions, and specific RAG evidence.
---
## 💻 API Usage Example
### Endpoint
`POST /analyze`
### Request (cURL)
```bash
curl -X 'POST' \
  'http://localhost:7860/analyze' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "type": "email",
  "subject": "Urgent: Update your account details",
  "sender": {
    "email": "security@baddomain.com"
  },
  "content": {
    "text": "Your account will be suspended if you do not click the link below to verify your identity."
  }
}'
```
### Expected Response
```json
{
  "classification": {
    "top_labels": [
      {
        "label": "phishing",
        "probability": 0.98
      }
    ],
    "model_version": "v1.0"
  },
  "risk_analysis": {
    "risk_score": 95,
    "risk_level": "critical",
    "reasons": [
      "Suspicious sender domain",
      "Urgency keywords detected"
    ],
    "model_version": "v1.0"
  },
  "recommended_actions": [
    "Do not click any links.",
    "Block the sender immediately."
  ],
  "rag_evidence": [
    "Similar phishing pattern detected in database on 2023-10-12."
  ],
  "metadata": {
    "processing_time_ms": 120,
    "models": {
      "classifier": "distilbert-base-uncased",
      "vector_store": "chromadb"
    }
  }
}
```
---
## 🛠️ Local Development
1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API server:**
   ```bash
   uvicorn app:app --reload --port 7860
   ```
4. **View Swagger Documentation:**
   Navigate to `http://localhost:7860/docs` to see the automated interactive OpenAPI interface.
