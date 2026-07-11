"""
Model 1 — Email Classifier.

This is the first stage of the pipeline. Its job is simple to state: given an
email, decide *what kind* of email it is (phishing, scam, spam, benign, ...).

We do this with "zero-shot classification": instead of training our own model on
labelled emails, we borrow a big pre-trained NLI model (BART-MNLI) that already
understands language, and ask it "how well does this email match the idea of
'phishing'? of 'spam'? ..." for each of our labels. No training data required.
"""
import time
import torch
from transformers import pipeline

from app.config.settings import settings
from app.utils.text_cleaner import clean_text

# The set of categories we want to sort every email into. Because this is
# zero-shot, we can add/remove labels here without retraining anything.
CANDIDATE_LABELS = [
    "phishing",
    "scam",
    "fraud",
    "malware",
    "spam",
    "benign"
]


class Model1Classifier:
    def __init__(self):
        self.model_name = settings.model1_name
        self.model_version = "model1-v1"

        # Pick the fastest hardware we have: Apple Silicon (mps) > NVIDIA (cuda)
        # > plain CPU. This lets the same code run on a laptop or a GPU box.
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Load the HuggingFace zero-shot pipeline once. This is heavy (downloads
        # the model on first run), which is why the class is created a single
        # time and reused for every request.
        self.hf_pipeline = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )

    def _run_inference(self, input_text: str) -> dict:
        """Run the model on one piece of text and return the ranked labels."""
        # `multi_label=False` runs a softmax across the labels so the
        # probabilities add up to 1.0 (treat the categories as mutually
        # exclusive). The hypothesis template is the sentence the model actually
        # scores, e.g. "This email is phishing." — good templates matter a lot.
        result = self.hf_pipeline(
            input_text,
            candidate_labels=CANDIDATE_LABELS,
            multi_label=False,
            hypothesis_template="This email is {}."
        )

        labels = result["labels"]
        scores = result["scores"]

        # --- Heuristic override for obvious short scams ---
        # Zero-shot models are surprisingly weak on tiny spammy texts like
        # "Congratulations! You won an iPhone!". When we spot those tell-tale
        # phrases, we force the phishing probability up to at least 0.65 and
        # rescale the remaining labels so everything still sums to 1.0.
        text_lower = input_text.lower()
        if "won an iphone" in text_lower or (
            "congratulations" in text_lower and ("won" in text_lower or "prize" in text_lower)
        ):
            target_phishing_prob = 0.65

            try:
                p_index = labels.index("phishing")
                current_p_prob = scores[p_index]
            except ValueError:
                p_index = -1
                current_p_prob = 0.0

            # Only bump it up if the model was under-confident to begin with.
            if current_p_prob < target_phishing_prob:
                # Distribute the leftover probability mass proportionally across
                # all the non-phishing labels so the total stays at 1.0.
                remaining_target = 1.0 - target_phishing_prob
                remaining_current = 1.0 - current_p_prob
                scale_factor = (remaining_target / remaining_current) if remaining_current > 0 else 0

                for i in range(len(scores)):
                    if labels[i] == "phishing":
                        scores[i] = target_phishing_prob
                    else:
                        scores[i] = scores[i] * scale_factor

        # The override may have changed the ranking, so re-sort highest-first.
        sorted_pairs = sorted(zip(labels, scores), key=lambda x: x[1], reverse=True)
        labels = [p[0] for p in sorted_pairs]
        scores = [p[1] for p in sorted_pairs]

        # Return the top 5 labels with clean, rounded probabilities.
        top_labels = []
        for label, score in list(zip(labels, scores))[:5]:
            top_labels.append({
                "label": label,
                "probability": round(float(score), 4)
            })

        return {"top_labels": top_labels}

    def predict(self, payload):
        start = time.perf_counter()

        # Pull out the subject and body defensively — any field may be missing.
        subject = getattr(payload, "subject", "") or ""

        text = ""
        html = ""
        if getattr(payload, "content", None):
            text = getattr(payload.content, "text", "") or ""
            html = getattr(payload.content, "html", "") or ""

        # Clean the body first (strip HTML tags, collapse whitespace) so the
        # model sees readable text, then combine subject + body into one string.
        body = clean_text(text=text, html=html)
        input_text = f"{subject} {body}".strip()
        if not input_text:
            # The model needs *something* to classify — never pass an empty string.
            input_text = "empty message"

        result_dict = self._run_inference(input_text)

        # Track how long this stage took (handy for latency debugging).
        elapsed = int((time.perf_counter() - start) * 1000)

        return {
            "top_labels": result_dict["top_labels"],
            "model_version": self.model_version,
            "time_ms": elapsed
        }
