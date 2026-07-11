"""
Render the Model-1 label probabilities as a horizontal bar chart.

Returns a base64-encoded PNG so the chart can be embedded directly in an
`<img src="data:image/png;base64,...">` tag or returned via the API without
writing any files to disk.
"""
import base64
import io
from typing import Dict, List

import matplotlib

# Use a non-interactive backend — the service runs headless (no display).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow backend selection)


def render_probability_chart(top_labels: List[Dict[str, float]]) -> str:
    """
    Build a bar chart from a list of {"label": str, "probability": float}
    entries and return it as a base64-encoded PNG string.

    Returns an empty string when there is nothing to plot.
    """
    if not top_labels:
        return ""

    labels = [item["label"] for item in top_labels]
    probs = [float(item["probability"]) for item in top_labels]

    # Highest probability on top.
    labels = labels[::-1]
    probs = probs[::-1]

    fig, ax = plt.subplots(figsize=(6, max(2, 0.5 * len(labels))))
    ax.barh(labels, probs, color="#4C72B0")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Classification confidence")

    for i, prob in enumerate(probs):
        ax.text(min(prob + 0.02, 0.95), i, f"{prob:.2f}", va="center")

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")
