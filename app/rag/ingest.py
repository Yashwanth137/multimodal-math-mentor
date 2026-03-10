from langchain_community.document_loaders import UnstructuredMarkdownLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.rag.vector_store import get_embeddings, save_vector_store
import os
from dotenv import load_dotenv

load_dotenv()

def ingest_docs(docs_dir: str, index_path: str):
    """
    Ingests markdown documents from the given directory into a FAISS index.
    """
    print(f"Ingesting documents from {docs_dir}...")
    
    # Load documents
    loader = DirectoryLoader(docs_dir, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Save vector store
    save_vector_store(vector_store, index_path)
    print(f"Vector store saved to {index_path}.")

if __name__ == "__main__":
    kb_dir = os.path.join("data", "knowledge_base")
    index_path = os.getenv("VECTOR_DB_PATH", "./app/rag/faiss_index")
    ingest_docs(kb_dir, index_path)
