import chromadb
from embeddings import LiteLLMEmbeddingAdapter

class VectorStoreManager:
    """Manages local in-memory ChromaDB operations and vector indexing."""
    def __init__(self, collection_name: str = "pdf_rag_knowledge_base"):
        self.client = chromadb.EphemeralClient()
        self.embedding_function = LiteLLMEmbeddingAdapter()
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

    def ingest_document_chunks(self, chunks: list[str]) -> None:
        """Indexes raw text segments into the collection database layer."""
        if not chunks:
            raise ValueError("Cannot ingest an empty list of text chunks.")
        
        string_ids = [f"id_chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, ids=string_ids)

    def search_relevant_context(self, question: str, top_k: int = 3) -> str:
        """Performs a geometric distance lookup and merges the top-k results into a single context string."""
        query_results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )
        
        # Extract matching document text strings list
        retrieved_docs = query_results.get('documents', [[]])[0]
        
        # Merge individual context chunks with clean spacing breaks
        return "\n\n---\n\n".join(retrieved_docs)
