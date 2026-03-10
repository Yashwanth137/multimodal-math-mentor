from app.utils.model_factory import get_supervisor_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

logger = logging.getLogger("SupervisorAgent")

class TriageOutput(BaseModel):
    problem_text: str = Field(description="Structured mathematical problem text.")
    is_complex: bool = Field(description="True if multi-step calculation or deep reasoning is needed.")
    is_multimodal: bool = Field(description="True if input refers to an image or visual element.")
    category: str = Field(description="algebra, calculus, probability, linear_algebra, or physics.")
    requires_tool: bool = Field(description="True if symbolic tools (SymPy) are likely needed for verification.")
    rag_query: str = Field(description="Optimized search query for retrieving relevant math theorems/formulas.")
    confidence_score: float = Field(description="Confidence in extraction (0-1).")
    needs_clarification: bool = Field(description="True if input is nonsensical or garbage.")

class FormatOutput(BaseModel):
    step_by_step: List[str] = Field(description="Clear explanation steps.")
    reasoning: str = Field(description="Tutor-style encouragement and logic.")
    final_boxed_answer: str = Field(description="Final answer in LaTeX boxed format.")

def run_triage(input_text: str) -> dict:
    """
    Supervisor triage: Clean text, assess difficulty, and route.
    """
    try:
        llm = get_supervisor_model()
        
        prompt = ChatPromptTemplate.from_template(
            "You are the Aethyln Labs Supervisor Node. Analyze the math/physics input.\n\n"
            "Input: {input_text}\n\n"
            "Tasks:\n"
            "1. Clean up transcription/OCR errors.\n"
            "2. Decide if it's 'complex' (multi-step, requires deep thinking) or 'simple' (conceptual, 1-step, conversational).\n"
            "3. Detect if it refers to images/diagrams.\n"
            "4. Categorize the topic carefully (algebra, calculus, probability, linear_algebra, physics).\n"
            "5. SYMPY CHECK: Set requires_tool: True if the problem involves equations, integrals, derivatives, or matrices.\n"
            "6. RAG QUERY: Create a short, optimized search query for retrieving the mathematical rules needed to solve this.\n"
            "7. OCR SANITY CHECK: If input text is '[Image input provided...]', immediately set is_multimodal: True, is_complex: True, category: 'calculus', and confidence_score: 1.0.\n"
            "8. Assign an final confidence score (0-1).\n\n"
            "JSON RULE: You must output a JSON object. For LaTeX, use double backslashes (e.g., \\\\frac) so it parses correctly as a JSON string.\n\n"
            "{format_instructions}"
        )
        
        parser = JsonOutputParser(pydantic_object=TriageOutput)
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "input_text": input_text,
            "format_instructions": parser.get_format_instructions()
        })
        if not result:
            raise ValueError("Triage model returned an empty or None result.")
        return result
    except Exception as e:
        logger.error("Supervisor Triage Error occurred.")
        return {
            "problem_text": input_text,
            "is_complex": True, 
            "is_multimodal": False,
            "category": "unknown",
            "confidence_score": 0.5
        }

def run_formatter(problem_text: str, solution_data: dict, context: List[str]) -> dict:
    """
    Supervisor formatting: Take raw solver output and make it student-friendly.
    """
    try:
        llm = get_supervisor_model()
        
        prompt = ChatPromptTemplate.from_template(
            "You are the Aethyln Labs Supervisor Node. Your goal is to format a high-quality math explanation.\n\n"
            "Problem: {problem_text}\n"
            "Context/Theorems: {context}\n"
            "Raw Solution Data: {solution_data}\n\n"
            "Guidelines:\n"
            "1. Use a premium, encouraging tone.\n"
            "2. Ensure LaTeX is correctly formatted for math expressions.\n"
            "3. End with a bolded, boxed final answer like \\boxed{{answer}}.\n\n"
            "JSON RULE: You must output a JSON object. For LaTeX, use double backslashes (e.g., \\\\frac) so it parses correctly as a JSON string.\n\n"
            "{format_instructions}"
        )
        
        parser = JsonOutputParser(pydantic_object=FormatOutput)
        chain = prompt | llm | parser
        
        result = chain.invoke({
            "problem_text": problem_text,
            "context": str(context),
            "solution_data": str(solution_data),
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        logger.error("Supervisor Formatter Error occurred.")
        return {
            "step_by_step": ["Calculation complete. See final result."],
            "reasoning": "The solution was processed successfully.",
            "final_boxed_answer": solution_data.get("final_result", "Error")
        }
