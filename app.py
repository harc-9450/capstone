from rag_pipeline import run_pipeline, search_similar_chunks
from models import generate_answer


version = input("📄 Choose document version (default: v1): ").strip() or "v1"
access = input("🔐 Enter your access level (All/HR/Legal): ").strip() or "All"

def main():
    
    print("📘 Welcome to the COBEC Policy Assistant (RAG-based)")

    while True:
        question = input("\n🧠 Ask a question (or type 'exit'): ")
        if question.lower() in ['exit', 'quit']:
            break
        
        # Step 1: Retrieve relevant chunks
        chunks = search_similar_chunks(question, top_k=3, version=version, access_level=access)
        if not chunks:
            print("⚠️ No matching content found for this version/access level.")
            continue

        # Step 2: Generate final answer using the Mistral model
        answer = generate_answer(question, chunks)

        print("\n💬 Answer:\n")
        print(answer)
        print("\n" + "-"*50 + "\n")



if __name__ == "__main__":
    main()