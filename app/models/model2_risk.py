"""
Model 2 — Risk Engine.

Stage two takes the email plus Model 1's verdict and turns it into a single
risk score (0-100) and a level (safe / suspicious / risky / high), along with a
list of human-readable reasons.

It's deliberately a *hybrid*: part machine-learning, part hand-written rules.
- The ML half (a zero-shot NLI model) spots fuzzy, language-based risk factors
  like "this asks for sensitive information".
- The rule half checks precise, explainable facts like "the sender's real domain
  doesn't match the domain they claim to be from".
Combining both gives us accuracy *and* an explanation for every point of risk.
"""
import time
import torch
from typing import Dict, Any
from transformers import pipeline

from app.config.settings import settings
from app.utils.text_cleaner import clean_text


class Model2RiskEngine:
    def __init__(self):
        # Use a proper NLI checkpoint here. The plain `deberta-v3-base` has no
        # entailment head, so it literally cannot do zero-shot classification —
        # this MNLI-fine-tuned version can.
        self.model_name = settings.model2_name
        self.model_version = "model2-v3"

        # Same device auto-detection as Model 1.
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Load the zero-shot model. If it fails to load for any reason we set
        # `analyzer = None` and carry on — the rule-based heuristics below still
        # work, so the service degrades gracefully instead of crashing.
        try:
            self.analyzer = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=device
            )
        except Exception as e:
            print(f"Could not load zero-shot pipeline for {self.model_name}: {e}. Falling back to default.")
            self.analyzer = None

        # The "risk factors" we ask the NLI model to check for in the body text.
        self.candidate_reasons = [
            "Contains urgent financial request",
            "Suspicious domain detected",
            "Requests sensitive information"
        ]

    def _run_nlp_analysis(self, text: str) -> list:
        """Ask the NLI model which risk factors apply to this text."""
        reasons = []
        if self.analyzer is not None and text.strip():
            try:
                result = self.analyzer(text, candidate_labels=self.candidate_reasons)
                # Only keep a risk factor when the model is fairly confident.
                # A lower threshold (e.g. 0.4) produced false positives such as
                # flagging "requests sensitive information" on a plain email.
                for label, score in zip(result['labels'], result['scores']):
                    if score > 0.65:
                        reasons.append(label)
            except Exception as e:
                print(f"Model 2 inference error: {e}")
        return reasons

    def analyze(self, payload: Any, model1_result: Dict) -> Dict[str, Any]:
        start = time.perf_counter()

        reasons = []
        risk_score = 0

        # Build one lowercase text blob (subject + cleaned body) for the NLP step.
        text = ""
        subject = ""
        if payload.content:
            text = clean_text(
                text=getattr(payload.content, "text", "") or "",
                html=getattr(payload.content, "html", "") or "",
            ).lower()
        if payload.subject:
            subject = payload.subject.lower()

        full_text = f"{subject} {text}".strip()

        # -------------------------------------------------------------------------
        # 1. Start from Model 1's classification.
        #    A confident "benign" keeps the score low; anything else is the main
        #    driver of risk (scaled by how confident Model 1 was).
        # -------------------------------------------------------------------------
        if model1_result and "top_labels" in model1_result and len(model1_result["top_labels"]) > 0:
            top_label = model1_result["top_labels"][0]["label"]
            top_prob = model1_result["top_labels"][0]["probability"]

            if top_label == "benign":
                # The more confident we are it's benign, the closer to 0.
                risk_score = max(0, 20 * (1 - top_prob))
            else:
                risk_score = top_prob * 60
                reasons.append(f"Message classified as potential {top_label}")

        # -------------------------------------------------------------------------
        # 2. Add the fuzzy, language-based risk factors from the NLI model.
        #    Each distinct factor adds a flat +10.
        # -------------------------------------------------------------------------
        nlp_reasons = self._run_nlp_analysis(full_text)
        for r in nlp_reasons:
            if r not in reasons:
                reasons.append(r)
                risk_score += 10

        # -------------------------------------------------------------------------
        # 3. Precise structural red flags (the "rules" half).
        # -------------------------------------------------------------------------
        # Attachments are a common malware vector.
        if payload.metadata and getattr(payload.metadata, "has_attachment", False):
            risk_score += 15
            reasons.append("Message contains attachment")

        # Links whose domain looks like a fake login/verification page
        # (e.g. "paypal-login-check.com") are classic phishing bait.
        if payload.links:
            for link in payload.links:
                domain = (getattr(link, "domain", "") or "").lower()
                for keyword in ["login", "verify", "secure", "update", "account"]:
                    if keyword in domain:
                        if "Suspicious verification-style domain detected" not in reasons:
                            risk_score += 15
                            reasons.append("Suspicious verification-style domain detected")
                        break

        # Sender spoofing: the domain in the actual "from" address doesn't match
        # the domain the sender claims to represent.
        if payload.sender and getattr(payload.sender, "email", None) and getattr(payload.sender, "domain", None):
            email_domain = payload.sender.email.split("@")[-1].lower()
            declared_domain = payload.sender.domain.lower()
            if email_domain != declared_domain:
                risk_score += 15
                reasons.append("Sender domain mismatch detected")

        # -------------------------------------------------------------------------
        # Finalize: cap the score at 100 and bucket it into a friendly level.
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

        # Belt-and-braces de-duplication of the reasons list.
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
