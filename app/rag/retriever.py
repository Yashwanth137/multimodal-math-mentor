from app.rag.vector_store import load_vector_store
from app.memory.memory_store import MemoryStore
import os
from dotenv import load_dotenv

load_dotenv()

# --- THE GLOBAL CACHE FIX ---
# These variables will hold the heavy models in memory so they only load once.
GLOBAL_VECTOR_STORE = None
GLOBAL_MEMORY_STORE = None

def get_global_vector_store(index_path=None):
    global GLOBAL_VECTOR_STORE
    if GLOBAL_VECTOR_STORE is None:
        path = index_path or os.getenv("VECTOR_DB_PATH", "./app/rag/faiss_index")
        # The heavy HuggingFace model is only loaded here on the very first run
        GLOBAL_VECTOR_STORE = load_vector_store(path)
    return GLOBAL_VECTOR_STORE

def get_global_memory_store():
    global GLOBAL_MEMORY_STORE
    if GLOBAL_MEMORY_STORE is None:
        GLOBAL_MEMORY_STORE = MemoryStore()
    return GLOBAL_MEMORY_STORE

class MathRetriever:
    """
    Enhanced retriever that pulls from both the Knowledge Base and Memory Store.
    """
    def __init__(self, index_path: str = None):
        # Simply point to the active memory slots instead of reloading!
        self.vector_store = get_global_vector_store(index_path)
        self.memory_store = get_global_memory_store()

    def retrieve(self, query: str, k: int = 3) -> list:
        """
        Retrieves the top k most relevant chunks across all sources.
        """
        combined_results = []
        
        # 1. Search Knowledge Base (Highest priority for formulas)
        if self.vector_store:
            docs = self.vector_store.similarity_search(query, k=k)
            for doc in docs:
                combined_results.append({
                    "content": f"Knowledge Base: {doc.page_content}",
                    "score": 0.9 # Default priority
                })
        
        # 2. Search Memory Store (Previous Solved Problems)
        memory_results = self.memory_store.semantic_search(query, k=2)
        for res in memory_results:
            combined_results.append({
                "content": f"Previously Solved (Memory ID {res['problem_id']}):\nProblem: {res['text']}\nSolution: {res['solution']}",
                "score": 0.7 # Lower priority for past cases unless highly relevant
            })
            
        # Sort and limit to top 3 total chunks
        return [res["content"] for res in combined_results[:k]]

if __name__ == "__main__":
    print("Retriever module initialized with Global Caching.")