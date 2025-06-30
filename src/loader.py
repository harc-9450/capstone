import os
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

def load_txt_documents(folder_path: str):
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            path = os.path.join(folder_path, file)
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = file
            documents.extend(docs)
    return documents