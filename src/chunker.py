from langchain_core.documents import Document

def convert_to_documents(flat_docs):
    """Converts list of dicts into LangChain Document format with metadata."""
    return [
        Document(page_content=entry["text"], metadata=entry["metadata"])
        for entry in flat_docs
    ]
