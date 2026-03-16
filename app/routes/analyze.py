# app/routes/analyze.py
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
    Body: AnalyzeRequest (validated by Pydantic)
    Returns: AnalyzeResponse (validated by Pydantic)
    """
    try:
        result = analyze_pipeline(payload)
        return result
    except Exception as e:
        logger.exception("Analyze pipeline failed: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error")