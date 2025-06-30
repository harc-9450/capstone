import os
import re
import streamlit as st
from rag_pipeline import search_similar_chunks
from models import generate_answer

st.set_page_config(page_title="RAG Research Assistant", layout="centered")
st.title("📘 Multi-Document RAG-Based Research Companion")

# Step 1: Detect all uploaded research paper PDFs
pdf_files = [f for f in os.listdir("docs") if f.endswith(".pdf")]
available_docs = [f.replace(".pdf", "") for f in pdf_files]

# Sidebar for filtering
st.sidebar.markdown("### 🔍 Document Filter")
selected_docs = st.sidebar.multiselect("Select documents to search", available_docs, default=available_docs)

# Main UI
user_question = st.text_input("🧠 Ask a research question:")

if user_question:
    with st.spinner("Retrieving and answering..."):
        chunks = search_similar_chunks(user_question, top_k=5, filter_docs=selected_docs)

        if not chunks:
            st.warning("⚠️ No relevant chunks found.")
        else:
            answer = generate_answer(user_question, chunks)
            st.markdown("### 💬 Answer")
            st.success(answer)

            with st.expander("📄 Retrieved Context Chunks"):
                for i, c in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}** — *{c['source']} / {c['section']}*")
                    st.code(c["chunk"])
