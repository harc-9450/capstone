# Q1 – GenAI Assignment: RAG-Based Movie QA System

### 🔍 Objective
Build a RAG-powered question answering system using:
- Sentence embeddings
- ChromaDB
- LangChain
- Gemini (LLM)

---

### 🧩 Pipeline Structure

1. **Load MovieLens data** (movies, users, ratings)
2. **Generate per-movie summaries** (rating, genre, demographics)
3. **Embed summaries** using different models (MiniLM, MPNet)
4. **Store in ChromaDB** for similarity search
5. **Answer questions** using LangChain + Gemini-Flash
6. **Evaluate accuracy** using BLEU, ROUGE, and EM

---

### 🔍 Sample Question

> *“Which sci-fi movie is highly rated by young adults?”*

**Answer (Gemini 2.0 Flash):**
> *It Came from Outer Space (1953) is viewed by 35 year olds and has a rating of 3.36.*

---

### 📊 Evaluation Scores (Selected)

| Question                                | BLEU  | ROUGE-1 | ROUGE-L | EM |
|-----------------------------------------|-------|----------|----------|----|
| Sci-fi movie for young adults           | 0.028 | 0.421    | 0.289    | 0  |
| Romantic comedy favored by females      | 0.007 | 0.133    | 0.133    | 0  |

---

### 🤖 Model Notes

- `gemini-1.5-flash-002`: Fast, but shallow reasoning
- `gemini-2.0-flash`: More complete, handles comparisons better
- `gemini-pro`: Inconsistent in LangChain (avoided)

---

### 🧠 Justification

We compared different embedding models and Gemini LLM variants. While BLEU/ROUGE helped score outputs, qualitative judgment was also applied. We selected `gemini-2.0-flash` due to its balanced performance and stable API support.

---

### ✅ Tools & Libraries

- LangChain v0.2+
- SentenceTransformers
- ChromaDB
- Gemini via `langchain_google_genai`
- Python 3.9.13