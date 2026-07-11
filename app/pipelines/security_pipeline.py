"""
Security pipeline — the orchestrator.

This module ties the whole system together. It owns the (heavy) model instances
and runs the four stages in order for each incoming email:

    Model 1 (classify)  ->  Model 2 (risk)  ->  RAG (evidence)  ->  Model 3 (actions)

The models are loaded once and cached in module-level globals, so we pay the
expensive load cost a single time rather than on every request.
"""
import os
import time

from app.config.settings import settings
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
# These start as None and get filled in on the first call to load_models().
model1 = None
model2 = None
model3 = None
rag_engine = None


def load_models():
    """Load the models once and reuse them forever.

    Safe to call repeatedly: after the first successful load, `model1` is no
    longer None so every later call is an instant no-op. We call this both at
    startup (to warm up) and at the top of analyze() (as a safety net).
    """
    global model1, model2, model3, rag_engine
    if model1 is None:
        # Make sure the vector store directory exists before Chroma tries to use it.
        os.makedirs(settings.vector_store_dir, exist_ok=True)
        model1 = Model1Classifier()
        model2 = Model2RiskEngine()
        model3 = Model3ActionGenerator()
        rag_engine = RAGEngine()
        # Build the RAG index from the knowledge base files.
        if hasattr(rag_engine, "rebuild_index"):
            rag_engine.rebuild_index()


def analyze(payload):
    """Run one email through all four stages and return the assembled response."""
    load_models()
    start_total = time.perf_counter()

    # -------------------------------------------------------------
    # 1. Classify the email (what kind of message is this?)
    # -------------------------------------------------------------
    model1_result = model1.predict(payload)

    # -------------------------------------------------------------
    # 2. Score the risk (how dangerous is it, and why?)
    #    Model 2 uses Model 1's output as one of its inputs.
    # -------------------------------------------------------------
    model2_result = model2.analyze(
        payload,
        model1_result
    )

    # -------------------------------------------------------------
    # 3. Retrieve supporting evidence from the knowledge base (RAG).
    #    We search using the email's subject + body as the query.
    # -------------------------------------------------------------
    text_content = payload.content.text if payload.content and payload.content.text else ""
    subject = payload.subject if payload.subject else ""
    query = f"{subject} {text_content}".strip()
    rag_evidence = rag_engine.get_evidence(query) if getattr(rag_engine, "get_evidence", None) else []

    # -------------------------------------------------------------
    # 4. Generate recommended actions, grounded in the risk + evidence.
    # -------------------------------------------------------------
    model3_result = model3.generate(
        {"risk_level": model2_result.get("risk_level", "unknown"), "reasons": model2_result.get("reasons", [])},
        rag_evidence
    )
    recommended_actions = model3_result.get("actions", [])

    # -------------------------------------------------------------
    # Record total processing time and which models we used.
    # Model names are read from the live instances so this can never drift
    # out of sync with what's actually loaded.
    # -------------------------------------------------------------
    total_time = int((time.perf_counter() - start_total) * 1000)

    metadata = Metadata(
        processing_time_ms=total_time,
        models={
            "model1": model1.model_name,
            "model2": model2.model_name,
            "model3": model3.model_name,
        }
    )

    # -------------------------------------------------------------
    # Assemble the validated response object and return it.
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
