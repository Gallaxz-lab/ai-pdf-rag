import os
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import litellm

class LiteLLMEmbeddingAdapter(EmbeddingFunction):
    """Forces ChromaDB to utilize our unified LiteLLM API wrapper for AI Studio."""
    def __init__(self, model_name: str = "gemini-embedding-2"):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        response = litellm.embedding(
            model=self.model_name,
            input=input,
            api_key=os.getenv("GEMINI_API_KEY"),
            custom_llm_provider="gemini"
        )
        return [item['embedding'] for item in response['data']]