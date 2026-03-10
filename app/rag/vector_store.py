from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def get_embeddings():
    """
    Streamlit caches this resource in background memory. 
    It survives page refreshes and will only execute ONCE per session!
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_vector_store(path: str = None):
    embeddings = get_embeddings()
    if path and os.path.exists(path):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None

def save_vector_store(vector_store, path: str):
    vector_store.save_local(path)

if __name__ == "__main__":
    print("Vector Store module initialized.")