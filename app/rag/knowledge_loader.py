"""
Knowledge-base loader.

Reads the cybersecurity guidance files (the source of truth for RAG) off disk
and turns them into LangChain Documents. Only plain-text formats (.txt / .md)
are picked up; anything else in the folder is ignored.
"""
from pathlib import Path
from typing import List, Optional, Union

from langchain_core.documents import Document

from app.config.settings import settings


def load_knowledge_base(directory: Optional[Union[str, Path]] = None) -> List[Document]:
    docs = []

    # Default to the configured knowledge-base directory when none is given.
    base_path = Path(directory) if directory is not None else settings.knowledge_base_dir

    # If the folder doesn't exist yet, just return an empty list (the RAG engine
    # knows how to handle "no documents").
    if not base_path.exists():
        return docs

    for file in base_path.glob("*"):

        # Skip non-text files (e.g. .DS_Store, __init__.py, binaries).
        if file.suffix not in [".txt", ".md"]:
            continue

        content = file.read_text(encoding="utf-8")

        # Keep the filename in metadata so we could trace evidence back to its
        # source document later if needed.
        docs.append(
            Document(
                page_content=content,
                metadata={"source": file.name}
            )
        )

    return docs
