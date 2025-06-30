# COBEC Policy Assistant (RAG-Based AI System)

This project is a GenAI-powered question-answering assistant designed to retrieve answers from COBEC policy documents based on version and access-level filters using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🔧 Tech Stack

- **LangChain** – Orchestration layer
- **Mistral-7B-Instruct** – Answer generation LLM
- **SentenceTransformer (MiniLM)** – Embedding model for document chunks
- **ChromaDB** – Vector store with persistent metadata support
- **Streamlit** – Frontend interface for interacting with the assistant

---

## 📁 Directory Structure

```
.
├── docs/                      # Folder to store all COBEC PDF documents
│   ├── COBECPolicy_v1_All.pdf
│   ├── COBECPolicy_v2_HR.pdf
│   └── COBECPolicy_v3_Legal.pdf
├── embeddings/chroma_db/     # ChromaDB persistent store
├── streamlit_app.py          # Main Streamlit UI
├── models.py                 # LLM loading and answer generation
├── rag_pipeline.py           # PDF processing and RAG pipeline
├── app.py                    # Terminal-based Q&A interface
└── README.md
```

---

## 🚀 How It Works

1. **PDF Parsing** – Files in `docs/` are parsed and chunked.
2. **Embedding** – Chunks are embedded using MiniLM and stored in ChromaDB.
3. **Version + Access Filtering** – Chunks are tagged with version and access-level metadata.
4. **Semantic Search** – Chunks matching the question and metadata are retrieved.
5. **LLM Answering** – Mistral generates a final answer using top-k relevant chunks.
6. **Frontend** – Users select version/access level and ask questions via Streamlit.

---

## 🧪 Supported File Naming Format

```
COBECPolicy_<version>_<access>.pdf

Examples:
COBECPolicy_v1_All.pdf
COBECPolicy_v2_HR.pdf
COBECPolicy_v3_Legal.pdf
```

---

## 🛠️ Run the App

### Step 1: Embed All Documents
```bash
python -c "from rag_pipeline import run_pipeline; run_pipeline()"
```

### Step 2: Launch Streamlit UI
```bash
streamlit run streamlit_app.py
```

---

## 📌 Smart Behavior

- Version filtering is strict (e.g., v1 ≠ v2).
- Access filtering supports fallback to `"All"` (e.g., HR users can also access `"All"` chunks).
- No chunk = no answer (with warning shown to user).

---

## ✅ Test Questions

### For HR (`v2`, `HR`)
- What actions does HCLTech take when an instance of child labor is reported?
- How does HCLTech ensure workplace safety and health for its employees?

### For Legal (`v3`, `Legal`)
- What rights and accommodations does HCL provide for employees with disabilities?
- How can an employee raise a complaint regarding discrimination under the Equal Opportunity Policy?

---

## 📈 Future Scope

- Upload support for new policies on UI
- Document comparison mode
- Hybrid search (BM25 + vector rerankers)

---

## 👤 Author

**Rohan Solanki**  
ID: 52033152