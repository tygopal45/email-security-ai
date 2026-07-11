"""
Embedding model wrapper.

Embeddings turn a piece of text into a list of numbers (a vector) that captures
its meaning, so texts about similar things end up close together in vector
space. The RAG engine uses these vectors to find relevant knowledge.

We use all-MiniLM-L6-v2: small, fast, and a well-proven default for RAG.
This thin wrapper just standardizes how the rest of the app creates it.
"""
from langchain_huggingface import HuggingFaceEmbeddings

from app.config.settings import settings


class EmbeddingModel:

    def __init__(self):
        self.model_name = settings.embedding_model_name

        # Downloads the model on first use, then caches it locally.
        self.embedding = HuggingFaceEmbeddings(
            model_name=self.model_name
        )

    def get(self):
        """Return the underlying LangChain embeddings object."""
        return self.embedding
