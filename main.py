import os
import io
import sys
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
import litellm


load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("Please set the GEMINI_API_KEY environment variable in your .env file.")
    sys.exit(1)

EMBEDDING_MODEL = "gemini-embedding-2"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Reads a local PDF file path and outputs a unified clean text string."""
    try:
        reader = PdfReader(pdf_path)
        extracted_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n".join(extracted_text).strip()
    except Exception as e:
        print(f"❌ Failed reading PDF file: {e}")
        sys.exit(1)
        
def split_text_into_chunks(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    """Splits the text into chunks of specified size with optional overlap."""
    chunks = []
    start_index = 0
    text_length = len(text)

    while start_index < text_length:
        end_index = start_index + chunk_size
        chunk = text[start_index:end_index]
        chunks.append(chunk.strip())
        
        start_index += (chunk_size - chunk_overlap)
        
    return [c for c in chunks if c]

class LiteLLMEmbeddingAdapter(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        response = litellm.embedding(
            model=EMBEDDING_MODEL,
            input=input,
            api_key=os.getenv("GEMINI_API_KEY"),
            custom_llm_provider="gemini"
        )
        return [item['embedding'] for item in response['data']]
    
def main():
    PDF_FILE_TARGET = "resume.pdf" 
    
    if not os.path.exists(PDF_FILE_TARGET):
        print(f"❌ Error: Please drop a sample PDF file named '{PDF_FILE_TARGET}' into your project folder.")
        print("You can copy any raw text resume, print it as a PDF, and save it as 'resume.pdf'.")
        return

    print("="*60)
    print("🚀 PIPELINE PHASE 1: PARSING AND PROCESSING")
    print("="*60)

    raw_document_text = extract_text_from_pdf(PDF_FILE_TARGET)
    print(f"✅ Text extracted successfully. Total character length: {len(raw_document_text)}")

    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 80
    text_chunks = split_text_into_chunks(raw_document_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    print(f"✅ Sliced text into {len(text_chunks)} distinct overlapping chunks.")
    print(f"👉 Parameters Active: Size={CHUNK_SIZE} chars | Overlap={CHUNK_OVERLAP} chars\n")

    # Display the chunks visually to observe formatting and boundary structures
    print("--- SAMPLE CHUNK ITERATION VISUALIZER ---")
    for idx, chunk in enumerate(text_chunks[:3]): # Displaying the first 3 slices
        print(f"📦 [CHUNK {idx + 1} / CHARACTER COUNT: {len(chunk)}]:")
        print(f"\"{chunk}\"")
        print("-" * 40)

    print("\n" + "="*60)
    print("💾 PIPELINE PHASE 2: VECTOR STORE EMBEDDING & STORAGE")
    print("="*60)

   
    chroma_client = chromadb.EphemeralClient()

    embedding_function = LiteLLMEmbeddingAdapter()
    collection = chroma_client.create_collection(
        name="resume_knowledge_store", 
        embedding_function=embedding_function
    )

    string_ids = [f"id_chunk_{i}" for i in range(len(text_chunks))]
    
    print("🧠 Generating mathematical thought vectors via LiteLLM...")
    collection.add(
        documents=text_chunks,
        ids=string_ids
    )
    print("✅ Vectors permanently mapped and indexed into ChromaDB collections storage layer.\n")

    print("="*60)
    print("🔍 PIPELINE PHASE 3: DIAGNOSTIC RETRIEVAL ANALYSIS")
    print("="*60)

    diagnostic_questions = [
        "What programming experience does this document mention?",
        "What programming languages are mentioned?",
        "What technical skills does the person have?"
    ]

    for rank, question in enumerate(diagnostic_questions, 1):
        print(f"\n❓ TEST QUESTION {rank}: \"{question}\"")
        print("⚡ Executing semantic neighborhood distance match...")
        
        query_results = collection.query(
            query_texts=[question],
            n_results=2
        )

        retrieved_documents = query_results['documents'][0]
        
        print("\n🎯 TOP SEMANTIC RETRIEVED CHUNKS FOR THIS CONTEXT:")
        print("-" * 50)
        for rank_idx, doc_chunk in enumerate(retrieved_documents, 1):
            print(f"🔹 Match Rank {rank_idx}:")
            print(f"\"{doc_chunk}\"")
            print("." * 40)
        print("-" * 50)

if __name__ == "__main__":
    main()