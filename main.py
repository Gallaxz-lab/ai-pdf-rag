import os
import sys
from dotenv import load_dotenv
from pdf_reader import extract_pages_from_pdf, split_page_data_into_chunks
from vector_store import DynamicVectorStoreManager
from llm import LLMManager
from prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE

load_dotenv()
TARGET_PDF = "resume.pdf"

if not os.path.exists(TARGET_PDF) or not os.getenv("GEMINI_API_KEY"):
    print("❌ Setup Error: Verify your .env token and ensure resume.pdf exists.")
    sys.exit(1)

def main():
    print("🚀 Extracting raw document structures page-by-page...")
    raw_pages = extract_pages_from_pdf(TARGET_PDF)
    
    # 🧪 EXPERIMENT SETUP: Segment configurations into 3 discrete size collections
    experiments = [200, 500, 1000]
    databases = {}
    
    for size in experiments:
        overlap = int(size * 0.15)  # Programmatic 15% sliding contextual overlap window
        chunks = split_page_data_into_chunks(raw_pages, chunk_size=size, chunk_overlap=overlap)
        
        vdb = DynamicVectorStoreManager(collection_name=f"db_size_{size}")
        vdb.ingest_structured_chunks(chunks)
        databases[size] = vdb
        print(f"📦 Ingested {len(chunks)} chunks into Collection Size={size} (Overlap={overlap})")

    ai_engine = LLMManager()

    print("\n" + "="*60)
    print("🧪 MULTI-SIZE CHUNK EXPERIMENT AND ATTRIBUTION BOT INITIALIZED")
    print("="*60 + "\n")

    while True:
        try:
            print("👤 You:")
            question = input(">> ").strip()
            if not question: continue
            if question.lower() in ["exit", "quit"]: break

            # Compare results side-by-side across all three configurations
            for size in experiments:
                print(f"\n⚙️ Running query pipeline inside Database Layer [ Size: {size} Chars ]...")
                
                # Perform search vector operation extracting context metadata parameters
                context, sources = databases[size].query_context_with_attribution(question, top_k=2)
                
                # Assemble system instructions context package payload tracking metrics
                payload = [
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": RAG_USER_TEMPLATE.format(context_text=context, user_question=question)}
                ]
                
                ai_answer = ai_engine.generate_chat_response(payload)
                
                print(f"🤖 Assistant [Size {size}]:")
                print(f"  {ai_answer}")
                print("📋 Source Attributions:")
                for s in sources:
                    print(f"  - Page {s['page']} | Chunk {s['chunk']}")
                print("-" * 50)
            print("\n" + "="*60)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
