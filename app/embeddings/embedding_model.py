from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    def __init__(self):

        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        self.embedding = HuggingFaceEmbeddings(
            model_name=self.model_name
        )

    def get(self):
        return self.embedding