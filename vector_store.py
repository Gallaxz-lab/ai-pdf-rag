"""
ChromaDB Management Layer with Metadata and Source Attribution Extraction.
"""
import chromadb
from embeddings import LiteLLMEmbeddingAdapter

class DynamicVectorStoreManager:
    """Manages ephemeral collection operations and formats sources for RAG contexts."""
    def __init__(self, collection_name: str):
        self.client = chromadb.EphemeralClient()
        self.embedding_function = LiteLLMEmbeddingAdapter()
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def ingest_structured_chunks(self, chunks_data: list[dict]) -> None:
        """Loads calculated lists into database arrays alongside metadata tags."""
        if not chunks_data:
            return
            
        documents = [c["text"] for c in chunks_data]
        ids = [c["chunk_id"] for c in chunks_data]
        metadatas = [c["metadata"] for c in chunks_data]
        
        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)

    def query_context_with_attribution(self, question: str, top_k: int = 2) -> tuple[str, list[dict]]:
        """Queries database and extracts matching text blocks alongside source tracking details."""
        query_results = self.collection.query(query_texts=[question], n_results=top_k)
        
        retrieved_docs = query_results.get('documents', [[]])[0]
        retrieved_meta = query_results.get('metadatas', [[]])[0]
        
        context_string = "\n\n---\n\n".join(retrieved_docs)
        
        # Compile a clean attribution array dictionary list for user printout routines
        attributions = [
            {"page": meta["page"], "chunk": meta["chunk_index"]} 
            for meta in retrieved_meta
        ]
        
        return context_string, attributions
