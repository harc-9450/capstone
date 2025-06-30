# 📘 Research Paper Companion (RAG Assistant) — Architecture & Flow

This document provides a conceptual and technical breakdown of how the **Research Paper Companion** works using Retrieval-Augmented Generation (RAG).

---

## ✅ Objective

Build an AI assistant capable of:
- Ingesting research PDFs
- Segmenting and embedding content
- Answering questions using an LLM grounded in document context
- Evaluating answer quality via metrics

---

## 🧩 Architecture Overview

```
               ┌────────────────────┐
               │ Research Paper PDF │
               └────────┬───────────┘
                        ▼
                ┌──────────────┐
                │ Chunking PDF │  ← PyMuPDF
                └─────┬────────┘
                      ▼
            ┌────────────────────┐
            │ Embedding Chunks   │  ← SentenceTransformers
            └───────┬────────────┘
                    ▼
            ┌────────────────────┐
            │ Store in Vector DB │  ← ChromaDB
            └────────┬───────────┘
                     ▼
┌──────────────────────────────┐
│ Ask Question via Streamlit   │
└────────────┬─────────────────┘
             ▼
   ┌────────────────────┐
   │ Retrieve Chunks    │  ← LangChain Retriever
   └────────┬───────────┘
            ▼
   ┌──────────────────────┐
   │ RAG Prompt + Gemini  │
   └────────┬─────────────┘
            ▼
   ┌──────────────────────┐
   │ Final Answer Output  │
   └──────────────────────┘
```

---

## 🧠 Components

### 📄 1. PDF Chunking (via `PyMuPDF`)
- Reads the paper page-by-page
- Chunks into 500–700 character segments
- Stores original page number for metadata

### 🔍 2. Embedding with BGE-Mini
- Each chunk is converted into a vector
- Model: `BAAI/bge-small-en` (or Gemini embedding API)
- Stored with metadata like source page

### 🧠 3. Vector Store (ChromaDB)
- Stores embeddings persistently
- Used for similarity search during retrieval

### 💬 4. LangChain RAG Pipeline
- User query → embedding → top-k similar chunks
- Chunks + question → sent to Gemini/Mistral LLM
- Answer is grounded in document context

---

## 🧪 Evaluation Module

- Uses pre-written test questions and gold answers
- Metrics:
  - BLEU
  - ROUGE-1, ROUGE-L
  - Exact Match

---

## 📊 Example Questions

- “What is the dataset used in the paper?”
- “How does the proposed method compare to X?”
- “What does the paper conclude about model performance?”

---

## 🔚 Outcome

- Demonstrates use of:
  - Retrieval-Augmented Generation (RAG)
  - LangChain + Gemini/Mistral integration
  - Custom data loader + vector store
- Supports research comprehension and literature surveys