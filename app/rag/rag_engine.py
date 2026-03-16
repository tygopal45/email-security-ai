# app/rag/rag_engine.py
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.embeddings.embedding_model import EmbeddingModel
from app.rag.knowledge_loader import load_knowledge_base


class RAGEngine:
    def __init__(self, persist_directory: str = "data/vector_store"):
        """
        Robust RAG engine:
         - filters out empty docs
         - avoids calling encode() with empty lists
         - creates an empty persistent Chroma store if there is nothing to index
        """
        self.embedding = EmbeddingModel().get()
        self.persist_directory = persist_directory
        self.vector_store: Optional[Chroma] = None

        # load docs (list[Document])
        docs: List[Document] = load_knowledge_base()

        # filter out empty documents (safeguard)
        docs = [d for d in docs if getattr(d, "page_content", "").strip()]

        if not docs:
            # create an empty/persistent Chroma store (no indexing)
            # this does not call embeddings.encode on an empty list
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
            return

        # split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # filter chunks with non-empty content
        chunks = [c for c in chunks if getattr(c, "page_content", "").strip()]

        if not chunks:
            # fallback to empty store
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
            return

        # Now safe to create vector store from non-empty chunks
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )

    def get_evidence(self, text: str, k: int = 3) -> List[str]:
        """
        Return up to k evidence snippets. If vector_store is None, return [].
        """
        if not text:
            return []

        if not self.vector_store:
            return []

        results = self.vector_store.similarity_search(text, k=k)  # returns Documents

        evidence = []
        for doc in results:
            snippet = getattr(doc, "page_content", "").strip()
            if snippet:
                evidence.append(snippet[:400])
        return evidence

    def rebuild_index(self, directory: str = "data/knowledge_base"):
        """
        Utility to rebuild the index from KB files (call when KB is updated).
        """
        docs = load_knowledge_base(directory)
        docs = [d for d in docs if getattr(d, "page_content", "").strip()]

        if not docs:
            # clear or create empty store
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