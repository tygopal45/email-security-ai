import time
import torch
from typing import Dict, Any
from langchain_core.runnables import RunnableLambda
from transformers import pipeline

class Model2RiskEngine:
    def __init__(self):
        self.model_name = "microsoft/deberta-v3-base"
        self.model_version = "model2-v3"

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Initialize the DeBERTa model for sequence classification to identify
        # specific risk factors inside the email body 
        try:
            self.analyzer = pipeline(
                "zero-shot-classification", 
                model=self.model_name, 
                device=device
            )
        except Exception as e:
            print(f"Could not load zero-shot pipeline for {self.model_name}: {e}. Falling back to default.")
            self.analyzer = None

        self.candidate_reasons = [
            "Contains urgent financial request",
            "Suspicious domain detected",
            "Requests sensitive information"
        ]

    def _run_nlp_analysis(self, text: str) -> list:
        reasons = []
        if self.analyzer is not None and text.strip():
            try:
                result = self.analyzer(text, candidate_labels=self.candidate_reasons)
                # Parse through probabilities
                for label, score in zip(result['labels'], result['scores']):
                    if score > 0.4:
                        reasons.append(label)
            except Exception as e:
                print(f"Model 2 inference error: {e}")
        return reasons

    def analyze(self, payload: Any, model1_result: Dict) -> Dict[str, Any]:
        start = time.perf_counter()

        reasons = []
        risk_score = 0

        # Construct input text string
        text = ""
        subject = ""
        if payload.content and payload.content.text:
            text = payload.content.text.lower()
        if payload.subject:
            subject = payload.subject.lower()
            
        full_text = f"{subject} {text}".strip()

        # -------------------------------------------------------------------------
        # 1. Model 1 Integration Score
        # -------------------------------------------------------------------------
        if model1_result and "top_labels" in model1_result and len(model1_result["top_labels"]) > 0:
            top_label = model1_result["top_labels"][0]["label"]
            top_prob = model1_result["top_labels"][0]["probability"]

            if top_label == "benign":
                # Inverse penalty if benign
                risk_score = max(0, 20 * (1 - top_prob))
            else:
                risk_score = top_prob * 60
                reasons.append(f"Message classified as potential {top_label}")

        # -------------------------------------------------------------------------
        # 2. DeBERTa NLP Analysis
        # -------------------------------------------------------------------------
        nlp_reasons = self._run_nlp_analysis(full_text)
        for r in nlp_reasons:
            if r not in reasons:
                reasons.append(r)
                risk_score += 10 # Bump risk for each NLP factor identified

        # -------------------------------------------------------------------------
        # 3. Structural Heuristics
        # -------------------------------------------------------------------------
        if payload.metadata and getattr(payload.metadata, "has_attachment", False):
            risk_score += 15
            reasons.append("Message contains attachment")

        # Link Analysis
        if payload.links:
            for link in payload.links:
                domain = (getattr(link, "domain", "") or "").lower()
                for keyword in ["login", "verify", "secure", "update", "account"]:
                    if keyword in domain:
                        if "Suspicious verification-style domain detected" not in reasons:
                            risk_score += 15
                            reasons.append("Suspicious verification-style domain detected")
                        break

        # Sender Mismatch
        if payload.sender and getattr(payload.sender, "email", None) and getattr(payload.sender, "domain", None):
            email_domain = payload.sender.email.split("@")[-1].lower()
            declared_domain = payload.sender.domain.lower()
            if email_domain != declared_domain:
                risk_score += 15
                reasons.append("Sender domain mismatch detected")

        # -------------------------------------------------------------------------
        # Clamp Risk Score and Classify Strategy
        # -------------------------------------------------------------------------
        risk_score = min(100, int(risk_score))

        if risk_score >= 75:
            risk_level = "high"
        elif risk_score >= 50:
            risk_level = "risky"
        elif risk_score >= 25:
            risk_level = "suspicious"
        else:
            risk_level = "safe"

        elapsed = int((time.perf_counter() - start) * 1000)

        # Ensure reasons list handles duplicates
        unique_reasons = []
        for r in reasons:
            if r not in unique_reasons:
                unique_reasons.append(r)

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reasons": unique_reasons,
            "model_version": self.model_version,
            "time_ms": elapsed,
        }