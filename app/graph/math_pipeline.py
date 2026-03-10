from typing import TypedDict, List, Optional, Union
from langgraph.graph import StateGraph, END
from app.agents.supervisor_agent import run_triage, run_formatter
from app.agents.solver_agent import run_solver_agent
from app.agents.verifier_agent import run_verifier_agent
from app.rag.retriever import MathRetriever
from app.memory.memory_store import MemoryStore
import os

# Define the state object for the graph
class AgentState(TypedDict):
    input_text: str
    image_path: Optional[str]
    triage_results: dict
    retrieved_context: List[str]
    solver_output: dict
    verification_results: dict
    retry_count: int
    final_explanation: dict
    status: str
    error_message: Optional[str]

# Node functions
def triage_node(state: AgentState):
    print("--- SUPERVISOR: TRIAGING INPUT ---")
    try:
        result = run_triage(state.get("input_text", "")) or {}
        if result.get("needs_clarification"):
            return {"triage_results": result, "status": "error", "error_message": "Input is unclear or nonsensical."}
        return {"triage_results": result, "status": "triaged", "retry_count": 0}
    except Exception as e:
        return {"status": "error", "error_message": f"Triage failed: {e}"}

def retriever_node(state: AgentState):
    print("--- SUPERVISOR: RETRIEVING CONTEXT ---")
    try:
        retriever = MathRetriever()
        triage = state.get("triage_results") or {}
        # Use structured rag_query from supervisor if available
        query = triage.get("rag_query") or triage.get("problem_text", state.get("input_text", ""))
        # category = triage.get("category") # Can be used for filtering
        context = retriever.retrieve(query)
        return {"retrieved_context": context, "status": "retrieved"}
    except Exception as e:
        return {"status": "error", "error_message": f"Retrieval failed: {e}"}

def solver_node(state: AgentState):
    print(f"--- SOLVER: DUAL-NODE ROUTING TO GEMINI (Attempt {state.get('retry_count', 0) + 1}) ---")
    try:
        triage_results = state.get("triage_results") or {}
        problem_text = triage_results.get("problem_text", state.get("input_text", ""))
        context = state.get("retrieved_context", [])
        image_path = state.get("image_path")
        
        # If this is a retry, append the previous critique
        if state.get("verification_results") and state["verification_results"].get("requires_retry"):
            critique = state["verification_results"].get("critique", "")
            problem_text += f"\n\nNOTE: A previous solution attempt failed verification with this critique: {critique}. Please fix these errors."

        result = run_solver_agent(problem_text, context, image_path)
        
        if result and result.get("status") == "error":
            return {"solver_output": result, "status": "error", "error_message": result.get("error_message")}
            
        return {"solver_output": result, "status": "solved", "retry_count": state.get("retry_count", 0) + 1}
    except Exception as e:
        return {"status": "error", "error_message": f"Solver failed: {e}"}

def verifier_node(state: AgentState):
    print("--- VERIFIER: CHECKING LOGICAL & SYMBOLIC CORRECTNESS ---")
    try:
        triage = state.get("triage_results") or {}
        problem_text = triage.get("problem_text", state.get("input_text", ""))
        solver_output = state.get("solver_output") or {}
        context = state.get("retrieved_context", [])
        
        result = run_verifier_agent(problem_text, solver_output, context)
        return {"verification_results": result, "status": "verified"}
    except Exception as e:
        return {"status": "error", "error_message": f"Verification failed: {e}"}

def formatting_node(state: AgentState):
    print("--- SUPERVISOR: FORMATTING FINAL OUTPUT ---")
    try:
        triage = state.get("triage_results") or {}
        problem_text = triage.get("problem_text", state.get("input_text", ""))
        context = state.get("retrieved_context", [])
        
        solution_data = state.get("solver_output") or {}
        
        # THE FRONTEND FIX: Provide dummy keys so the Streamlit UI doesn't crash!
        if not solution_data:
            solution_data = {
                "steps": ["Problem solved instantly by Supervisor (Fast-Lane Routing)."],
                "final_answer": "See final explanation.",
                "reasoning_summary": "Skipped heavy solver for simple conceptual query.",
                "tools_used_detail": [],
                "status": "solved"
            }
        
        if state.get("status") == "error":
            return {"status": "error"}
            
        result = run_formatter(problem_text, solution_data, context)
        
        # YOU MUST RETURN 'solver_output' HERE SO THE UI SEES THE DUMMY DATA
        return {
            "final_explanation": result, 
            "solver_output": solution_data, 
            "status": "completed"
        }
    except Exception as e:
        return {"status": "error", "error_message": f"Formatting failed: {e}"}
        
# Routing logic
def route_after_triage(state: AgentState):
    if state["status"] == "error": return "end"
    return "continue"

def route_after_retrieval(state: AgentState):
    if state.get("status") == "error": return "end"
    triage = state.get("triage_results") or {}
    if triage.get("is_complex") or triage.get("is_multimodal") or state.get("image_path"):
        return "solver"
    return "formatter"

def route_after_solver(state: AgentState):
    if state["status"] == "error": return "end"
    return "continue"

def route_after_verification(state: AgentState):
    if state["status"] == "error": return "end"
    v = state["verification_results"]
    if v.get("requires_retry") and state.get("retry_count", 0) < 2: # Max 2 retries
        print(f"!!! VERIFICATION FAILED: Routing back to Solver (Count: {state['retry_count']}) !!!")
        return "retry"
    return "finalize"

# Build the graph
def create_math_pipeline():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("solver", solver_node)
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("formatter", formatting_node)

    # Add Edges
    workflow.set_entry_point("triage")
    
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "continue": "retriever",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "retriever",
        route_after_retrieval,
        {
            "solver": "solver",
            "formatter": "formatter",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "solver",
        route_after_solver,
        {
            "continue": "verifier",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "retry": "solver",
            "finalize": "formatter",
            "end": END
        }
    )
    
    workflow.add_edge("formatter", END)

    return workflow.compile()

def run_pipeline(input_text: str, image_path: str = None):
    app = create_math_pipeline()
    initial_state = {
        "input_text": input_text,
        "image_path": image_path,
        "triage_results": {},
        "retrieved_context": [],
        "solver_output": {},
        "verification_results": {},
        "retry_count": 0,
        "final_explanation": {},
        "status": "started"
    }
    return app.invoke(initial_state)

if __name__ == "__main__":
    print("Math Pipeline Graph initialized.")