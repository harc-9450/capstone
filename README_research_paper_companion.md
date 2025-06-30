# 🧠 Research Paper Companion (RAG-Based Assistant)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to assist with understanding and querying research papers (PDFs). It uses document chunking, embeddings, and a local LLM to answer queries based on academic content.

---

## 🚀 Features

- ✅ Accepts PDF research papers (e.g., arXiv, IEEE)
- ✅ Uses LangChain & ChromaDB for RAG pipeline
- ✅ LLM-based answers grounded in document context
- ✅ Gemini or Mistral LLM support (configurable)
- ✅ Streamlit-based interface for user interaction
- ✅ Evaluation via BLEU, ROUGE, and Exact Match

---

## 🛠️ Tech Stack

- Python 3.9
- PyMuPDF (fitz) for PDF parsing
- LangChain
- ChromaDB
- SentenceTransformers (BGE-Mini or others)
- Gemini or HuggingFace LLMs (e.g., Mistral)
- Streamlit (optional for UI)

---

## 📂 Folder Structure

```
research_paper_companion_RAG/
├── app.py                # Streamlit UI
├── models.py             # Load LLM + embedding model
├── rag_pipeline.py       # RAG logic: chunk, embed, query
├── evaluate.py           # Model evaluation
├── test_cases.py         # Sample questions + gold answers
├── Research_Paper.pdf    # Sample input
└── embeddings/           # ChromaDB persistent vector store
```

---

## ✅ Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run Streamlit interface:
```bash
streamlit run app.py
```

3. Ask questions like:
```text
What dataset was used in this paper?
How does the proposed method compare to baseline X?
```

---

## 📊 Sample Output

```
Query: What model is proposed in the paper?

Answer: The paper proposes a vision-language transformer pre-trained with masked modeling.
```

---

## 📌 Credits
Inspired by LangChain RAG best practices and powered by open LLMs.