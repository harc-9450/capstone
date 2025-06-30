from evaluate import evaluate_answer
from qa_pipeline import answer_question

test_cases = [
    {
        "question": "Which sci-fi movie is highly rated by young adults?",
        "expected": "It Came from Outer Space (1953) is a highly rated sci-fi movie watched by users aged 35."
    },
    {
        "question": "Which romantic comedy is favored by female users?",
        "expected": "Notting Hill is a romantic comedy rated highly by female viewers."
    }
]

model_name = "all-MiniLM-L6-v2"
llm_model = "gemini-1.5-flash-002"
db_path = f"embeddings/chroma_db_{model_name.replace('/', '_')}"

for case in test_cases:
    print(f"🔍 Question: {case['question']}")
    predicted = answer_question(case["question"], model_name, db_path, llm_model=llm_model)
    print(f"🤖 Answer: {predicted}")
    print(f"✅ Expected: {case['expected']}")

    scores = evaluate_answer(predicted, case['expected'])
    print("📊 Scores:", scores)
    print("-" * 60)
