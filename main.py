import os
import sys
from dotenv import load_dotenv
from pdf_reader import extract_text_from_pdf, split_text_into_chunks
from vector_store import VectorStoreManager
from llm import LLMManager
from prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE

load_dotenv()

def bootstrap_knowledge_base() -> VectorStoreManager:
    """Initializes the database and parses the local PDF file content."""
    TARGET_PDF = "resume.pdf"
    
    if not os.path.exists(TARGET_PDF):
        print(f"❌ Error: Missing mandatory document file source target '{TARGET_PDF}'.")
        print("Please place your target PDF file inside this repository folder root before running.")
        sys.exit(1)
        
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Critical Error: GEMINI_API_KEY environment token not found in .env.")
        sys.exit(1)

    print("⚙️  Parsing and indexing local document structure data...")
    
    # 1. Initialize DB storage structure wrapper
    db_manager = VectorStoreManager()
    
    # 2. Process physical file chunks text extraction
    raw_text = extract_text_from_pdf(TARGET_PDF)
    text_chunks = split_text_into_chunks(raw_text, chunk_size=500, chunk_overlap=80)
    
    # 3. Save calculated vectors into indexing layer
    db_manager.ingest_document_chunks(text_chunks)
    print(f"✅ Ingestion complete. {len(text_chunks)} document chunks indexed successfully.\n")
    
    return db_manager

def main():
    # Initialize your component architectural layers
    db = bootstrap_knowledge_base()
    ai_engine = LLMManager()

    print("==========================================================")
    print("📄  PDF CONTEXTUAL AI ASSISTANT CLIENT LOOP CHATBOT       ")
    print("==========================================================")
    print(" -> Core Status: Operational")
    print(" -> Semantic Query Context Guardrails: Active")
    print(" -> Commands: Type 'exit' or 'quit' to close connection.")
    print("==========================================================\n")

    # Initialize your conversation tracker list.
    # The system prompt enforces strict rules at index 0.
    chat_history = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT}
    ]

    while True:
        try:
            print("👤 You:")
            user_question = input(">> ").strip()
            
            if not user_question:
                continue
            if user_question.lower() in ["exit", "quit"]:
                print("\n👋 Severing backend session. Goodbye!")
                break

            print("\n🔍 Fetching knowledge base facts...")
            # Step A: Perform vector lookup using Top-K (fetching top 3 relevant chunks)
            matched_context = db.search_relevant_context(user_question, top_k=3)

            # Step B: Build your context-injected user prompt string composition
            augmented_user_prompt = RAG_USER_TEMPLATE.format(
                context_text=matched_context,
                user_question=user_question
            )

            # Step C: Append the newly engineered payload block to history list
            chat_history.append({"role": "user", "content": augmented_user_prompt})

            print("🤖 AI Thinking...")
            # Step D: Route the entire conversational context payload block to the model
            ai_answer = ai_engine.generate_chat_response(chat_history)

            print("\n🤖 Assistant:")
            print("-" * 60)
            print(ai_answer)
            print("-" * 60 + "\n")

            # Step E: Append the clean assistant string answer to preserve conversation state
            chat_history.append({"role": "assistant", "content": ai_answer})

        except KeyboardInterrupt:
            print("\n\n👋 Forced connection drop. Goodbye!")
            break

if __name__ == "__main__":
    main()
