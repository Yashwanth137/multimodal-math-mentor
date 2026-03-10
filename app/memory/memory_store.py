import sqlite3
import json
import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional
from app.rag.vector_store import get_embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import logging
logger = logging.getLogger("MemoryStore")

load_dotenv()

class MemoryStore:
    """
    Enhanced persistent memory store for JEE math problems.
    Supports SQLite for relational data and semantic vector search via FAISS.
    """
    def __init__(self, db_path: str = None, vector_store_path: str = None):
        if db_path is None:
            db_path = os.getenv("DATABASE_URL", "./app/memory/memory.db")
            if db_path.startswith("sqlite:///"):
                db_path = db_path.replace("sqlite:///", "")
        
        if vector_store_path is None:
            vector_store_path = os.getenv("MEMORY_VECTOR_PATH", "./app/memory/faiss_memory")

        self.db_path = db_path
        self.vector_store_path = vector_store_path
        self.embeddings = get_embeddings()
        self._init_db()
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self) -> Optional[FAISS]:
        """Loads the semantic vector store from disk if it exists."""
        if os.path.exists(self.vector_store_path):
            try:
                return FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                # Log error or handle gracefully
                return None
        return None

    def _init_db(self):
        """Initializes the SQLite database with expanded schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT UNIQUE,
                original_input TEXT,
                parsed_problem TEXT,
                retrieved_context TEXT,
                solution TEXT,
                steps TEXT,
                final_answer TEXT,
                reasoning_summary TEXT,
                debug_trace TEXT,
                verification_status TEXT,
                user_feedback TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migration: Add new columns if missing
        cursor.execute("PRAGMA table_info(memory)")
        columns = [col[1] for col in cursor.fetchall()]
        new_cols = {
            "problem_id": "TEXT",
            "steps": "TEXT",
            "final_answer": "TEXT",
            "reasoning_summary": "TEXT",
            "debug_trace": "TEXT"
        }
        for col, col_type in new_cols.items():
            if col not in columns:
                cursor.execute(f"ALTER TABLE memory ADD COLUMN {col} {col_type}")

        conn.commit()
        conn.close()

    def check_duplicate(self, problem_text: str) -> Optional[str]:
        """Checks if a problem already exists and returns its problem_id."""
        
        # FIX: Never deduplicate image file paths!
        if problem_text.endswith(('.png', '.jpg', '.jpeg')):
            return None
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT problem_id FROM memory WHERE original_input = ? LIMIT 1", (problem_text,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def add_memory(self, original_input: str, parsed_problem: dict, retrieved_context: List[str], 
                   solution: str, verification_status: str, steps: Optional[List[str]] = None,
                   final_answer: Optional[str] = None, reasoning_summary: Optional[str] = None,
                   debug_trace: Optional[dict] = None, user_feedback: Optional[str] = None):
        """
        Adds a new record to SQLite and updates the semantic vector index.
        Prevents duplicates by checking problem text first.
        """
        # Check for existing
        existing_id = self.check_duplicate(original_input)
        if existing_id:
            logger.info(f"Duplicate problem detected. Returning existing ID: {existing_id}")
            return existing_id

        import uuid
        problem_id = str(uuid.uuid4())
        
        # 1. Save to SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO memory (problem_id, original_input, parsed_problem, retrieved_context, 
                               solution, steps, final_answer, reasoning_summary, debug_trace,
                               verification_status, user_feedback)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            problem_id,
            original_input,
            json.dumps(parsed_problem),
            json.dumps(retrieved_context),
            solution,
            json.dumps(steps) if steps else None,
            final_answer,
            reasoning_summary,
            json.dumps(debug_trace) if debug_trace else None,
            verification_status,
            user_feedback
        ))
        conn.commit()
        conn.close()

        # 2. Update Vector Store
        doc = Document(
            page_content=original_input,
            metadata={
                "problem_id": problem_id, 
                "solution": solution,
                "category": parsed_problem.get("category", "math")
            }
        )
        if self.vector_store:
            self.vector_store.add_documents([doc])
        else:
            self.vector_store = FAISS.from_documents([doc], self.embeddings)
        
        self.vector_store.save_local(self.vector_store_path)
        return problem_id

    def semantic_search(self, query: str, k: int = 3) -> List[dict]:
        """Performs semantic vector search over past problems."""
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return [
            {
                "problem_id": res.metadata["problem_id"],
                "text": res.page_content,
                "solution": res.metadata.get("solution"),
                "category": res.metadata.get("category")
            }
            for res in results
        ]

    def get_all_history(self) -> List[dict]:
        """Retrieves summary of all past problems for the UI sidebar."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, problem_id, original_input, timestamp FROM memory ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def clear_all(self):
        """Deletes all memory records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memory")
        conn.commit()
        conn.close()
        # Optionally reset vector store too
        if os.path.exists(self.vector_store_path):
            import shutil
            shutil.rmtree(self.vector_store_path, ignore_errors=True)
            self.vector_store = None

    def get_full_solution(self, problem_id: str) -> Optional[dict]:
        """Retrieves the full record for a specific problem."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memory WHERE problem_id = ?", (problem_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            res = dict(row)
            res["parsed_problem"] = json.loads(res["parsed_problem"]) if res.get("parsed_problem") else {}
            res["retrieved_context"] = json.loads(res["retrieved_context"]) if res.get("retrieved_context") else []
            res["steps"] = json.loads(res["steps"]) if res.get("steps") else []
            res["debug_trace"] = json.loads(res["debug_trace"]) if res.get("debug_trace") else {}
            return res
        return None

if __name__ == "__main__":
    print("Memory Store module initialized.")
