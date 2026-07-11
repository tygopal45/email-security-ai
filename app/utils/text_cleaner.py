"""
Text cleaning utilities.

Email bodies frequently arrive as HTML with tracking markup, inline styles and
irregular whitespace. Feeding that raw into the transformer models hurts both
latency and classification quality, so callers normalize text here first.
"""
import re

from bs4 import BeautifulSoup

_WHITESPACE_RE = re.compile(r"\s+")


def strip_html(html: str) -> str:
    """Return the visible text from an HTML fragment (scripts/styles removed)."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and trim."""
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str = "", html: str = "") -> str:
    """
    Produce a single clean text string for model input.

    Uses `text` when present; otherwise falls back to the visible text of
    `html`. The result always has normalized whitespace.
    """
    source = text if (text and text.strip()) else strip_html(html)
    return normalize_whitespace(source)
