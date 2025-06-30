
# 📄 Q13: RAG-Based Assistant Using `.txt` Files (LangChain + Gemini)

## ✅ Problem Statement

> Use LangChain and Retrieval Augmented Generation (RAG) to answer a question using multiple `.txt` documents connected to a large language model (LLM). You can use OpenAI or any other LLM such as Gemini.

---

## 💡 Summary

This project builds a RAG-based AI assistant that:
- Loads multiple `.txt` files
- Embeds and stores them in a vector database
- Uses Gemini LLM to answer questions using relevant file content

---

## 🧱 Tech Stack

| Component      | Tool / Service                     |
|----------------|------------------------------------|
| LLM            | Gemini (`gemini-1.5-flash`)        |
| Embedding      | `models/embedding-001` (Gemini)    |
| Vector Store   | ChromaDB                           |
| Framework      | LangChain                          |
| Format         | CLI + `.txt` files                 |

---

## 📁 Folder Structure

```
q13_rag_text_assistant/
├── data/                    # ⬅️ Contains .txt files
├── src/
│   ├── loader.py            # Load and label each .txt file
│   ├── embed_store.py       # Embed + store in ChromaDB
│   ├── query_engine.py      # Gemini + RetrievalQA
├── main.py                  # CLI app to ask questions
├── .env                     # GOOGLE_API_KEY
```

---

## 🚀 How It Works

1. **Load all .txt files** using LangChain TextLoader
2. **Convert to Documents** with metadata (`source_file`)
3. **Embed** using Gemini’s embedding model
4. **Store in ChromaDB** locally
5. **Use RetrievalQA** with Gemini to answer questions
6. **Print Gemini response + matching document snippets**

---

## 📄 Sample Questions to Try

Based on the 20 Newsgroups content:
- “What is a Bricklin sports car?”
- “How do I change icons in Windows 3.0?”
- “What do users think about SCSI drives?”
- “What treatment did someone mention for brain tumors?”
- “Was DiskDoubler compatible with Autodoubler?”

---

## ✅ Deliverables

- [x] Load and label multiple .txt files
- [x] Embed with Gemini and store in Chroma
- [x] Ask questions using RetrievalQA
- [x] Show relevant document snippets with filenames

---

## 🔐 Setup

1. Install:
```bash
pip install langchain langchain-google-genai chromadb python-dotenv
```

2. Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key
```

3. Run:
```bash
python main.py
```

