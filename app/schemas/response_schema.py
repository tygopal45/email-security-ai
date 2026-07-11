"""
Response schema — the shape of what POST /analyze returns.

Declaring the route's `response_model` as AnalyzeResponse means FastAPI both
validates and filters our output: we can only ever return exactly these fields,
which keeps the API predictable and prevents accidentally leaking internals.
"""
from pydantic import BaseModel, Field
from typing import List, Dict


class LabelProbability(BaseModel):
    label: str            # e.g. "phishing"
    probability: float    # 0.0 - 1.0


class Classification(BaseModel):
    """Model 1's output: the ranked category guesses."""
    top_labels: List[LabelProbability] = Field(default_factory=list)
    model_version: str = ""


class RiskAnalysis(BaseModel):
    """Model 2's output: the numeric verdict and why."""
    risk_score: int = 0            # 0 - 100
    risk_level: str = "safe"       # safe / suspicious / risky / high
    reasons: List[str] = Field(default_factory=list)
    model_version: str = ""


class Metadata(BaseModel):
    """Diagnostics about the run itself."""
    processing_time_ms: int = 0
    models: Dict[str, str] = Field(default_factory=dict)   # which models ran


class AnalyzeResponse(BaseModel):
    classification: Classification          # what kind of email
    risk_analysis: RiskAnalysis             # how risky + reasons
    recommended_actions: List[str] = Field(default_factory=list)   # what to do
    rag_evidence: List[str] = Field(default_factory=list)          # supporting knowledge
    metadata: Metadata
