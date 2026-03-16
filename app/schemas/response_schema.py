from pydantic import BaseModel, Field
from typing import List, Dict


class LabelProbability(BaseModel):
    label: str
    probability: float


class Classification(BaseModel):
    top_labels: List[LabelProbability] = Field(default_factory=list)
    model_version: str = ""


class RiskAnalysis(BaseModel):
    risk_score: int = 0
    risk_level: str = "safe"
    reasons: List[str] = Field(default_factory=list)
    model_version: str = ""


class Metadata(BaseModel):
    processing_time_ms: int = 0
    models: Dict[str, str] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    classification: Classification
    risk_analysis: RiskAnalysis
    recommended_actions: List[str] = Field(default_factory=list)
    rag_evidence: List[str] = Field(default_factory=list)
    metadata: Metadata