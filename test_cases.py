from evaluate import evaluate_answer
from models import generate_answer
from rag_pipeline import search_similar_chunks

# Sample test
test_cases = [
    {
        "question": "What programming languages are associated with artificial intelligence?",
        "expected": "LISP and Prolog"
    },
    {
        "question": "What does the paper say about the future of AI?",
        "expected": "The role of humans will change as machines begin doing their work"
    }
]

for test in test_cases:
    print(f"\n🧠 Question: {test['question']}")
    chunks = search_similar_chunks(test["question"], top_k=3)
    generated = generate_answer(test["question"], chunks)
    print(f"🤖 Answer: {generated}")
    print(f"✅ Reference: {test['expected']}")

    scores = evaluate_answer(generated, test["expected"])
    print("📊 Evaluation:", scores)
