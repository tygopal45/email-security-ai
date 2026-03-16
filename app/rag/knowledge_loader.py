from pathlib import Path
from langchain_core.documents import Document
from typing import List


def load_knowledge_base(directory: str = "data/knowledge_base") -> List[Document]:

    docs = []

    base_path = Path(directory)

    if not base_path.exists():
        return docs

    for file in base_path.glob("*"):

        if file.suffix not in [".txt", ".md"]:
            continue

        content = file.read_text(encoding="utf-8")

        docs.append(
            Document(
                page_content=content,
                metadata={"source": file.name}
            )
        )

    return docs