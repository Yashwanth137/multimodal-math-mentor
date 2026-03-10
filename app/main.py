from app.graph.math_pipeline import run_pipeline as run_graph_pipeline, create_math_pipeline
from app.memory.memory_store import MemoryStore
from app.ocr.ocr_pipeline import run_ocr
from app.asr.speech_to_text import run_asr
from app.hitl.hitl_manager import HITLManager
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("Main")

def ingest_rag():
    """
    Utility to ingest documents into the RAG vector store.
    """
    from app.rag.ingest import ingest_docs
    kb_dir = os.path.join("data", "knowledge_base")
    index_path = os.getenv("VECTOR_DB_PATH", "./app/rag/faiss_index")
    ingest_docs(kb_dir, index_path)

def process_math_input(input_data: str, mode: str = "text", ui_status=None):
    """
    Main orchestration function to process image, audio, or text math problems.
    Optimized for Aethyln Labs Dual-Node architecture.
    """
    hitl_manager = HITLManager()
    extraction_result = {"text": input_data, "confidence": 1.0, "status": "success"}
    image_path = None

    # 1. Input Processing
    logger.info(f"Processing input in mode: {mode}")
    if mode == "image":
        image_path = input_data
        extraction_result = {
            "text": "[Image input provided. Analyze the visual for mathematical content.]",
            "confidence": 1.0,
            "status": "success"
        }
    elif mode == "audio":
        extraction_result = run_asr(input_data)

    # 2. HITL Check for Extraction
    hitl_eval = hitl_manager.evaluate_extraction(extraction_result, mode)
    if hitl_eval["needs_hitl"]:
        logger.warning(f"HITL Triggered: {hitl_eval['reason']}")
        return {
            "status": "needs_hitl",
            "reason": hitl_eval["reason"],
            "raw_text": extraction_result["text"],
            "confidence": extraction_result["confidence"]
        }

    # 3. Run Pipeline
    if ui_status:
        app = create_math_pipeline()
        initial_state = {
            "input_text": extraction_result["text"],
            "image_path": image_path,
            "triage_results": {},
            "retrieved_context": [],
            "solver_output": {},
            "verification_results": {},
            "retry_count": 0,
            "final_explanation": {},
            "status": "started"
        }
        final_state = initial_state
        for event in app.stream(initial_state):
            for node_name, state in event.items():
                final_state = state
                if node_name == "triage": ui_status.write("✅ Triaged problem and extracted intent...")
                elif node_name == "retriever": ui_status.write("📚 Retrieved context from Knowledge Base...")
                elif node_name == "solver": ui_status.write("⚙️ Solver Agent derived step-by-step logic...")
                elif node_name == "verifier": ui_status.write("🔍 Verifier Agent checked mathematical correctness...")
                elif node_name == "formatter": ui_status.write("📝 Formatted final explanation...")
    else:
        final_state = run_graph_pipeline(extraction_result["text"], image_path=image_path)
    
    # 4. Secondary HITL Check
    triage = final_state.get("triage_results", {})
    if triage.get("needs_clarification"):
        return {
            "status": "needs_hitl",
            "reason": "ocr_garbage_detected",
            "raw_text": triage.get("problem_text", extraction_result["text"]),
            "confidence": triage.get("confidence_score", 0)
        }

    # 5. Handle Errors
    if final_state.get("status") == "error":
        return {
            "status": "error",
            "error_message": final_state.get("error_message", "Unknown pipeline error"),
            "debug": final_state
        }

    # 6. Save to Memory
    memory = MemoryStore()
    solver_data = final_state.get("solver_output", {})
    explanation = final_state.get("final_explanation", {})
    
    problem_id = memory.add_memory(
        original_input=input_data,
        parsed_problem=triage,
        retrieved_context=final_state.get("retrieved_context"),
        solution=explanation.get("final_boxed_answer", "N/A"),
        verification_status=str(final_state.get("verification_results", {}).get("is_correct", True)),
        steps=explanation.get("step_by_step") or solver_data.get("steps"),
        final_answer=explanation.get("final_boxed_answer") or solver_data.get("final_answer"),
        reasoning_summary=explanation.get("reasoning") or solver_data.get("reasoning_summary"),
        debug_trace={
            "triage": triage,
            "retrieval": final_state.get("retrieved_context"),
            "solver": solver_data,
            "verifier": final_state.get("verification_results")
        }
    )

    return {
        "problem_id": problem_id,
        "explanation": explanation,
        "parsed_problem": triage,
        "category": triage.get("category", "math"),
        "retrieved_context": final_state.get("retrieved_context"),
        "solution": {
            "steps": solver_data.get("steps"),
            "final_answer": solver_data.get("final_answer"),
            "reasoning_summary": solver_data.get("reasoning_summary"),
            "tools_used": solver_data.get("tools_used_detail", [])
        },
        "verification": final_state.get("verification_results", {}),
        "status": final_state.get("status"),
        "debug": {
            "triage": triage,
            "retrieval": final_state.get("retrieved_context"),
            "solver": solver_data,
            "verifier": final_state.get("verification_results")
        }
    }

def get_session_history():
    return MemoryStore().get_all_history()

def get_past_solution(problem_id: str):
    record = MemoryStore().get_full_solution(problem_id)
    if not record: return None
    return {
        "problem_id": problem_id,
        "explanation": {
            "step_by_step": record.get("steps") or ["Archive: Steps not captured."],
            "final_boxed_answer": record.get("final_answer") or record.get("solution"),
            "reasoning": record.get("reasoning_summary") or "No reasoning available."
        },
        "parsed_problem": record.get("parsed_problem"),
        "category": record.get("parsed_problem", {}).get("category", "math"),
        "retrieved_context": record.get("retrieved_context"),
        "solution": {
            "steps": record.get("steps"),
            "final_answer": record.get("final_answer"),
            "reasoning_summary": record.get("reasoning_summary"),
            "tools_used": (record.get("debug_trace") or {}).get("solver", {}).get("tools_used_detail", [])
        },
        "verification": {
            "is_correct": record.get("verification_status") == "True",
            "critique": ((record.get("debug_trace") or {}).get("verifier") or {}).get("critique", "No critique available.")
        },
        "status": "completed",
        "debug": record.get("debug_trace")
    }

def clear_session_history():
    MemoryStore().clear_all()

if __name__ == "__main__":
    print("Main App Logic initialized.")
