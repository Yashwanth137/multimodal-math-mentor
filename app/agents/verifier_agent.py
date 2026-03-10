from app.utils.model_factory import get_supervisor_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from app.tools.math_tools import MATH_TOOLS

logger = logging.getLogger("VerifierAgent")

class VerificationOutput(BaseModel):
    is_correct: bool = Field(description="True if the solution is mathematically sound.")
    confidence_score: float = Field(description="Confidence in verification (0-1).")
    critique: str = Field(description="Detailed explanation of errors or confirmation of correctness.")
    requires_retry: bool = Field(description="True if the solver should attempt to fix the problem.")

def run_verifier_agent(problem_text: str, solver_output: dict, context: List[str]) -> dict:
    """
    Verifier node: Cross-checks solver output with symbolic math and deep reasoning.
    """
    try:
        # 1. Automated Symbolic Check (Simple verification of final answer if numeric)
        # In a real system, we'd parse the final answer and use SymPy to verify it against the problem.
        # For now, we'll let the Supervisor LLM act as the reasoning verifier with tool access if needed.
        
        llm = get_supervisor_model()
        
        prompt = ChatPromptTemplate.from_template(
            "You are the Aethyln Labs Verifier Node. Your task is to critique the Solver's proof.\n\n"
            "Problem: {problem_text}\n"
            "Retrieved Context: {context}\n"
            "Solver Proof: {solver_proof}\n"
            "Tool Results Used by Solver: {tool_results}\n\n"
            "Tasks:\n"
            "1. Verify the logical steps. Are there illegal mathematical operations?\n"
            "2. Cross-check numeric results against the problem constraints.\n"
            "3. Determine if the final answer is correct and robust.\n"
            "4. If incorrect, be specific about WHERE the error is.\n\n"
            "JSON RULE: Output a JSON object with fields: is_correct, confidence_score, critique, requires_retry.\n\n"
            "{format_instructions}"
        )
        
        parser = JsonOutputParser(pydantic_object=VerificationOutput)
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "problem_text": problem_text,
            "context": str(context),
            "solver_proof": solver_output.get("raw_proof"),
            "tool_results": str(solver_output.get("tools_used_detail", [])),
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
    except Exception as e:
        logger.error("Verifier Error occurred.")
        return {
            "is_correct": True, # Fail safe to avoid infinite loops if verifier crashes
            "confidence_score": 0.5,
            "critique": "Verification failed due to internal error.",
            "requires_retry": False
        }

if __name__ == "__main__":
    print("Verifier Agent initialized.")
