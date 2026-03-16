# app/pipeline/security_pipeline.py
import time

from app.models.model1_classifier import Model1Classifier
from app.models.model2_risk import Model2RiskEngine
from app.models.model3_action import Model3ActionGenerator
from app.rag.rag_engine import RAGEngine

from app.schemas.response_schema import (
    AnalyzeResponse,
    Classification,
    RiskAnalysis,
    Metadata,
)

# -------- LAZY MODEL LOADING --------

model1 = None
model2 = None
model3 = None
rag_engine = None

def load_models():
    global model1, model2, model3, rag_engine
    if model1 is None:
        import os
        os.makedirs("data/vector_store", exist_ok=True)
        model1 = Model1Classifier()
        model2 = Model2RiskEngine()
        model3 = Model3ActionGenerator()
        rag_engine = RAGEngine()
        # Optionally ensure index is loaded
        if hasattr(rag_engine, "rebuild_index"):
            rag_engine.rebuild_index()


def analyze(payload):
    load_models()
    start_total = time.perf_counter()

    # -------------------------------------------------------------
    # 1. RUN MODEL 1 (SPAM / INTENT CLASSIFICATION)
    # -------------------------------------------------------------
    model1_result = model1.predict(payload)

    # -------------------------------------------------------------
    # 2. RUN MODEL 2 (RISK ENGINE)
    # -------------------------------------------------------------
    model2_result = model2.analyze(
        payload,
        model1_result
    )

    # -------------------------------------------------------------
    # 3. RAG RETRIEVAL (EVIDENCE GATHERING)
    # -------------------------------------------------------------
    text_content = payload.content.text if payload.content and payload.content.text else ""
    subject = payload.subject if payload.subject else ""
    query = f"{subject} {text_content}".strip()
    rag_evidence = rag_engine.get_evidence(query) if getattr(rag_engine, "get_evidence", None) else []

    # -------------------------------------------------------------
    # 4. RUN MODEL 3 (ACTION GENERATION) 
    # -------------------------------------------------------------
    model3_result = model3.generate(
        {"risk_level": model2_result.get("risk_level", "unknown"), "reasons": model2_result.get("reasons", [])},
        rag_evidence
    )
    recommended_actions = model3_result.get("actions", [])

    # -------------------------------------------------------------
    # TOTAL LATENCY METRICS
    # -------------------------------------------------------------
    total_time = int((time.perf_counter() - start_total) * 1000)

    metadata = Metadata(
        processing_time_ms=total_time,
        models={
            "model1": "facebook/bart-large-mnli",
            "model2": "microsoft/deberta-v3-base",
            "model3": "google/flan-t5-base"
        }
    )

    # -------------------------------------------------------------
    # BUILD RESPONSE
    # -------------------------------------------------------------
    response = AnalyzeResponse(
        classification=Classification(
            top_labels=model1_result["top_labels"],
            model_version=model1_result["model_version"]
        ),
        risk_analysis=RiskAnalysis(
            risk_score=model2_result["risk_score"],
            risk_level=model2_result["risk_level"],
            reasons=model2_result["reasons"],
            model_version=model2_result["model_version"]
        ),
        recommended_actions=recommended_actions,
        rag_evidence=rag_evidence,
        metadata=metadata
    )

    return response