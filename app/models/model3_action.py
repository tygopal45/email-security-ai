import time
import logging
from typing import List, Dict, Any, Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from transformers import pipeline

logger = logging.getLogger("email-security-ai.model3")
logger.addHandler(logging.NullHandler())

class Model3ActionGenerator:
    """
    Generates user safety actions using google/flan-t5-base
    via LangChain LCEL implementation
    """
    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "cpu"):
        self.model_name = model_name
        self.model_version = "model3-v2"
        
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

        # Build prompt template
        self.prompt = PromptTemplate.from_template(
            "Based on a {risk_level} email risk with these reasons: {reasons_text}. "
            "Relevant cybersecurity knowledge: {evidence_text}. "
            "What should the user do? List 3 specific short actionable steps separated by commas."
        )

    def _run_generation(self, prompt_text: str) -> str:
        if self.generator is None:
            return ""
        try:
            result = self.generator(
                prompt_text.text if hasattr(prompt_text, "text") else str(prompt_text),
                max_new_tokens=60,
                do_sample=False,
                num_return_sequences=1,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            return result[0]['generated_text']
        except Exception as e:
            logger.exception("Model 3 generation failed: %s", e)
            return ""

    def _parse_output(self, generated_text: str) -> List[str]:
        if not generated_text:
            return []
        
        # Split by both commas and periods to break up run-on and repeating sentences
        raw_actions = [a.strip() for a in generated_text.replace(".", ",").split(",") if a.strip()]
        
        # Deduplicate while preserving order
        actions = []
        for a in raw_actions:
            if a.lower() not in [existing.lower() for existing in actions]:
                actions.append(a)
                
        return actions[:5]

    def _fallback_actions(self) -> List[str]:
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
        
        reasons_text = ", ".join(reasons) if reasons else "None"
        evidence_text = " ".join(rag_evidence) if rag_evidence else "None"

        # Format prompt using LangChain and execute
        prompt_value = self.prompt.format(
            risk_level=risk_level, 
            reasons_text=reasons_text, 
            evidence_text=evidence_text
        )
        actions = self._parse_output(self._run_generation(prompt_value))

        if not actions:
            actions = self._fallback_actions()

        elapsed = int((time.perf_counter() - start) * 1000)

        return {
            "actions": actions,
            "model_version": self.model_version,
            "time_ms": elapsed
        }