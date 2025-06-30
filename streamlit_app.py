import os
import re
import streamlit as st
from rag_pipeline import search_similar_chunks
from models import generate_answer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="HCL Policy Knowledge Assistant", layout="wide")

# === Sidebar ===
with st.sidebar:
    st.title("🛠 Settings")
    st.markdown("Choose your document version and access level.")
    
    pdf_files = [f for f in os.listdir("docs") if f.endswith(".pdf") and "COBECPolicy" in f]
    version_pattern = re.compile(r"COBECPolicy_(v\d+)_", re.IGNORECASE)
    available_versions = sorted({match.group(1) for f in pdf_files if (match := version_pattern.search(f))})
    st.sidebar.write("🔍 Extracted Versions:", available_versions)
    if not available_versions:
        available_versions = ["v1"]  # fallback

    selected_version = st.selectbox("📄 Select Policy Version", available_versions)
    access_level = st.radio("🔐 Access Level", ["All", "HR", "Legal"], index=0)

# === Main Area ===
st.title("📘 HCL Policy Knowledge Assistant")
st.markdown("Use this tool to ask questions grounded in official HCL policy documents.")

with st.container():
    st.markdown("### 🔍 Ask a Question")
    user_question = st.text_input("Type your query about the HCL policy below:", placeholder="e.g., What are the leave policies for HR staff?")

    if user_question:
        with st.spinner("💡 Searching documents and thinking..."):
            chunks = search_similar_chunks(user_question, top_k=3, version=selected_version, access_level=access_level)

            if not chunks:
                st.warning("⚠️ No relevant content found for this version and access level.")
            else:
                answer = generate_answer(user_question, chunks)
                st.success("✅ Answer generated successfully!")
                st.markdown("### 💬 Answer:")
                st.info(answer)

                with st.expander("📚 Show Retrieved Context"):
                    for i, chunk in enumerate(chunks):
                        st.markdown(f"**Chunk {i+1}**")
                        st.code(chunk, language="markdown")