"""
Model 3 — Action Generator.

The final stage turns the analysis into plain advice for the user: "here are 3
things you should do about this email". It uses google/flan-t5-base, a small
instruction-following text-to-text model (you give it an instruction, it writes
a response).

We feed it the risk level, the reasons, and the evidence retrieved by the RAG
step, so its advice is grounded in real context rather than made up.
"""
import time
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from transformers import pipeline

from app.config.settings import settings

logger = logging.getLogger("email-security-ai.model3")
logger.addHandler(logging.NullHandler())


class Model3ActionGenerator:
    """Generates short, actionable user safety steps with google/flan-t5-base."""

    def __init__(self, model_name: Optional[str] = None, device: str = "cpu"):
        self.model_name = model_name or settings.model3_name
        self.model_version = "model3-v2"

        # Load the text-generation model. As with Model 2, a load failure isn't
        # fatal: `generator` becomes None and we fall back to canned advice later.
        try:
            self.generator = pipeline(
                task="text2text-generation",
                model=self.model_name,
                device=device
            )
            logger.info("Model3 loaded successfully: %s", self.model_name)
        except Exception as e:
            logger.exception("Failed to load Model3: %s", e)
            self.generator = None

        # The instruction we give the model. The placeholders get filled in with
        # the actual risk level, reasons, and RAG evidence at request time.
        self.prompt = PromptTemplate.from_template(
            "Based on a {risk_level} email risk with these reasons: {reasons_text}. "
            "Relevant cybersecurity knowledge: {evidence_text}. "
            "What should the user do? List 3 specific short actionable steps separated by commas."
        )

    def _run_generation(self, prompt_text: str) -> str:
        """Run the model on the filled-in prompt and return the raw text."""
        if self.generator is None:
            return ""
        try:
            result = self.generator(
                prompt_text.text if hasattr(prompt_text, "text") else str(prompt_text),
                max_new_tokens=60,
                do_sample=False,        # greedy decoding -> deterministic output
                num_return_sequences=1,
                # These two stop a small model from getting stuck repeating
                # itself ("delete the message delete the message ...").
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            return result[0]['generated_text']
        except Exception as e:
            logger.exception("Model 3 generation failed: %s", e)
            return ""

    def _parse_output(self, generated_text: str) -> List[str]:
        """Turn the model's free-form text into a clean list of steps."""
        if not generated_text:
            return []

        # The model separates steps with commas, but sometimes runs on with
        # periods too — split on both, then drop blanks.
        raw_actions = [a.strip() for a in generated_text.replace(".", ",").split(",") if a.strip()]

        # Remove duplicates (case-insensitive) while keeping the original order.
        actions = []
        for a in raw_actions:
            if a.lower() not in [existing.lower() for existing in actions]:
                actions.append(a)

        return actions[:5]

    def _fallback_actions(self) -> List[str]:
        """Safe, sensible defaults used when the model produces nothing usable."""
        return [
            "Do not click any links in the message.",
            "Verify the sender through the official website.",
            "Report the message as phishing.",
            "Delete the suspicious message."
        ]

    def generate(self, risk_analysis: Dict[str, Any], rag_evidence: List[str]) -> Dict[str, Any]:
        start = time.perf_counter()

        risk_level = risk_analysis.get("risk_level", "unknown")
        reasons = risk_analysis.get("reasons", [])

        # "None" is a deliberate literal string here — it reads fine inside the
        # prompt sentence when there are no reasons/evidence to report.
        reasons_text = ", ".join(reasons) if reasons else "None"
        evidence_text = " ".join(rag_evidence) if rag_evidence else "None"

        # Fill the template, run the model, and parse the result.
        prompt_value = self.prompt.format(
            risk_level=risk_level,
            reasons_text=reasons_text,
            evidence_text=evidence_text
        )
        actions = self._parse_output(self._run_generation(prompt_value))

        # Never return an empty, useless list — fall back to safe defaults.
        if not actions:
            actions = self._fallback_actions()

        elapsed = int((time.perf_counter() - start) * 1000)

        return {
            "actions": actions,
            "model_version": self.model_version,
            "time_ms": elapsed
        }
