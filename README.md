
# 📊 Q2: RAG-Based Excel Assistant using LangChain + Gemini

## ✅ Problem Statement

> Implement a RAG system for extracting information from multiple Excel sheets using LLM, Langchain, word embedding, excel sheet prompt and others tools if necessary. If possible display the extracted information in a table format.

---

## 🧾 Dataset Used

- **File**: `Canada.xlsx`
- **Source**: Public Canadian immigration dataset
- **Structure**: Multiple sheets with tabular data on immigration statistics

---

## ⚙️ Tech Stack

| Component       | Tool/Library                        |
|----------------|-------------------------------------|
| Vector DB       | ChromaDB                           |
| Embeddings      | Google Gemini `models/embedding-001` |
| LLM             | Gemini (`gemini-1.5-flash`)         |
| Framework       | LangChain                          |
| Format          | CLI output with `pandas` tables     |

---

## 📁 Folder Structure

```
q2_rag_excel_assistant/
├── data/
│   └── Canada.xlsx
├── src/
│   ├── data_loader.py       # Flatten Excel to dict
│   ├── chunker.py           # LangChain Document wrapper
│   ├── embed_store.py       # Embed and store to Chroma
│   ├── query_engine.py      # Gemini-based RetrievalQA
├── chroma_db/               # Persisted vector store
├── main.py                  # CLI query app
├── .env                     # GOOGLE_API_KEY
```

---

## ✅ Features & Workflow

1. **Load & Flatten Excel**  
   All sheets are loaded and each row is converted to structured text.

2. **Chunk & Convert to LangChain Documents**  
   Each row includes `page_content` + `metadata` (`sheet`, `row`).

3. **Embedding with Gemini**  
   Documents are embedded using Gemini’s embedding model (`embedding-001`).

4. **Stored in ChromaDB**  
   Vector store persisted on disk (`chroma_db/`).

5. **Query via Gemini LLM**  
   Questions are answered using `RetrievalQA` with context-aware LLM generation.

6. **Table Output**  
   Matching rows are shown in table format using `pandas.to_markdown()`.

---

## 🧪 Example Output

```
❓ Enter your question about the Excel data:
How many immigrants came to Canada from India in 2005?

🤖 Gemini Answer:
In 2005, 36210 immigrants came to Canada from India.

📄 Top Matching Chunks:
📝 Sheet: Canada by Citizenship (2), Row: 79
| Country | Year | Immigrants |
|---------|------|------------|
| India   | 2005 | 36210      |
```

---

## 🧼 Optimizations

- Reuses ChromaDB if already created
- Uses Gemini for both embedding and answering
- Provides fallback if `tabulate` is missing

---

## ✅ Final Deliverables

- [x] Load + flatten Excel with multiple sheets
- [x] Embed & store in Chroma using Gemini
- [x] Retrieval-based QA with Gemini LLM
- [x] CLI question + answer system
- [x] Matched rows shown in table format

---

## 📎 Setup

```bash
pip install -r requirements.txt
```

Create `.env`:
```
GOOGLE_API_KEY=your_api_key_here
```

Run:
```bash
python main.py
```
