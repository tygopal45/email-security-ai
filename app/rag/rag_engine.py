# app/rag/rag_engine.py
"""
RAG engine — Retrieval-Augmented Generation.

The idea: rather than hoping the action-generating model already "knows" good
security advice, we keep that advice in a knowledge base of text files, and at
request time we *retrieve* the most relevant snippets to feed into the prompt.

How it works:
  1. Load the knowledge-base documents.
  2. Split them into small overlapping chunks.
  3. Turn each chunk into a vector (embedding) and store it in Chroma.
  4. At query time, embed the query and find the closest chunks by similarity.

This class is written defensively so an empty or missing knowledge base never
crashes the service — it just returns no evidence.
"""
from pathlib import Path
from typing import List, Optional, Union

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config.settings import settings
from app.embeddings.embedding_model import EmbeddingModel
from app.rag.knowledge_loader import load_knowledge_base


class RAGEngine:
    def __init__(self, persist_directory: Optional[Union[str, Path]] = None):
        """Build (or open) the vector store from the knowledge base."""
        self.embedding = EmbeddingModel().get()
        # Where Chroma persists its index on disk. Defaults to the configured
        # vector-store directory but can be overridden (handy for tests).
        self.persist_directory = str(
            persist_directory if persist_directory is not None else settings.vector_store_dir
        )
        self.vector_store: Optional[Chroma] = None

        # Load the knowledge-base files as Documents.
        docs: List[Document] = load_knowledge_base()

        # Drop any empty documents — embedding empty text is pointless and some
        # backends error on it.
        docs = [d for d in docs if getattr(d, "page_content", "").strip()]

        if not docs:
            # Nothing to index yet: create an empty (but valid) persistent store
            # so get_evidence() can run without blowing up.
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
            return

        # Split documents into ~500-char chunks with a little overlap, so search
        # returns focused passages and we don't cut sentences off mid-thought.
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Same empty-content safeguard, now at the chunk level.
        chunks = [c for c in chunks if getattr(c, "page_content", "").strip()]

        if not chunks:
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
            return

        # Embed every chunk and build the searchable vector store.
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )

    def get_evidence(self, text: str, k: Optional[int] = None) -> List[str]:
        """Return up to `k` knowledge-base snippets most relevant to `text`."""
        # Fall back to the configured default (3) when no k is passed.
        if k is None:
            k = settings.rag_top_k

        # Empty query or no index -> nothing to return.
        if not text:
            return []

        if not self.vector_store:
            return []

        # Fetch a few extra candidates so we can drop duplicates / noise and
        # still return up to `k` useful snippets.
        results = self.vector_store.similarity_search(text, k=k * 2)

        evidence = []
        seen = set()
        for doc in results:
            snippet = getattr(doc, "page_content", "").strip()
            if not snippet:
                continue

            # Skip chunks that are just a markdown heading (e.g. "# Email Threat
            # Types") — they add no real information.
            if self._is_heading_only(snippet):
                continue

            snippet = snippet[:400]

            # De-duplicate: the same chunk can surface more than once.
            key = snippet.strip().lower()
            if key in seen:
                continue
            seen.add(key)

            evidence.append(snippet)
            if len(evidence) >= k:
                break

        return evidence

    @staticmethod
    def _is_heading_only(snippet: str) -> bool:
        """True if the snippet is a single markdown heading line with no body."""
        lines = [ln for ln in snippet.splitlines() if ln.strip()]
        return len(lines) == 1 and lines[0].lstrip().startswith("#")

    def rebuild_index(self, directory: Optional[Union[str, Path]] = None):
        """Re-read the knowledge base and rebuild the index from scratch.

        Call this after the knowledge base files change so the new content is
        searchable without restarting the app.
        """
        docs = load_knowledge_base(directory if directory is not None else settings.knowledge_base_dir)
        docs = [d for d in docs if getattr(d, "page_content", "").strip()]

        if not docs:
            # No docs -> reset to an empty store.
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        chunks = [c for c in chunks if getattr(c, "page_content", "").strip()]

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
