# app/routes/analyze.py
"""
The /analyze route.

This is intentionally thin: its only job is to accept a request, hand it to the
pipeline, and return the result. FastAPI + Pydantic handle input validation for
us before this function ever runs, so bad input becomes an automatic 422.
"""
from fastapi import APIRouter, HTTPException
import logging

from app.schemas.request_schema import AnalyzeRequest
from app.schemas.response_schema import AnalyzeResponse
from app.pipelines.security_pipeline import analyze as analyze_pipeline

logger = logging.getLogger("email-security-ai.routes")

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze_route(payload: AnalyzeRequest):
    """
    POST /analyze

    Body:    an email described by AnalyzeRequest (validated by Pydantic).
    Returns: the full security assessment as AnalyzeResponse.
    """
    try:
        result = analyze_pipeline(payload)
        return result
    except Exception as e:
        # Log the real error for us, but return a generic 500 to the caller so
        # we never leak internal details (stack traces, model errors, etc.).
        logger.exception("Analyze pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")
