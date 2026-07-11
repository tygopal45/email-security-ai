"""
Central configuration for Email Security AI.

Paths are anchored to the `app/` package directory so the service behaves the
same regardless of the current working directory (important on HF Spaces where
uvicorn may be launched from an arbitrary location).
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .../app/config/settings.py -> parents[1] == .../app
APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"


class Settings(BaseSettings):
    # --- Paths (absolute, CWD-independent) ---
    knowledge_base_dir: Path = DATA_DIR / "knowledge_base"
    vector_store_dir: Path = DATA_DIR / "vector_store"

    # --- Model identifiers ---
    model1_name: str = "facebook/bart-large-mnli"
    model2_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    model3_name: str = "google/flan-t5-base"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- RAG ---
    rag_top_k: int = 3

    model_config = SettingsConfigDict(
        env_prefix="EMAIL_SEC_",
        extra="ignore",
    )


settings = Settings()
