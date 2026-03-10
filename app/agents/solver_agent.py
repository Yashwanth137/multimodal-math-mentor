import os
import time
import base64
import logging
from typing import List, Optional
from app.utils.model_factory import get_solver_model
from app.tools.math_tools import MATH_TOOLS
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

logger = logging.getLogger("SolverAgent")

class SolverOutput(BaseModel):
    steps: List[str] = Field(description="Step-by-step mathematical proof or derivation.")
    final_answer: str = Field(description="The final result, formatted in LaTeX.")
    reasoning_summary: str = Field(description="High-level summary of the logic and key theorems used.")
    tools_used: List[str] = Field(description="Names of symbolic tools used during the solution.")
    raw_proof: str = Field(description="The full, unformatted proof for debugging purposes.")

def encode_image(image_path: str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_solver_agent(problem_text: str, context: list, image_path: str = None) -> dict:
    """
    Elite reasoning node (Gemini) with SymPy tool-calling and instant ResourceExhausted fallback.
    """
    # Define tools for LLM
    tools = [
        {"name": "sympy_solver", "description": "Solves a symbolic math expression or equation.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
        {"name": "derivative_solver", "description": "Computes the derivative.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}, "variable": {"type": "string"}}, "required": ["expression"]}},
        {"name": "integral_solver", "description": "Computes integrals.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}, "variable": {"type": "string"}, "lower": {"type": "number"}, "upper": {"type": "number"}}, "required": ["expression"]}},
        {"name": "simplify_expression", "description": "Simplifies expression.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}
    ]

    parser = JsonOutputParser(pydantic_object=SolverOutput)
    
    system_msg = (
        "You are the JEE Math Professor. Your objective is elite mathematical reasoning.\n"
        "1. If an image is attached, YOU MUST READ AND ANALYZE THE IMAGE to find the math problem.\n"
        "2. USE SYMPY TOOLS for any complex calculations, derivatives, or integrals.\n"
        "3. PROVIDE STEP-BY-STEP PROOFS.\n"
        "4. OUTPUT MUST BE VALID JSON.\n\n"
        f"{parser.get_format_instructions()}"
    )
    
    content = [
        {
            "type": "text", 
            "text": f"CRITICAL INSTRUCTION: An image of a math problem is attached. You must read it and solve it. \n\nOCR/Text Hint: {problem_text}\nContext: {str(context)}"
        }
    ]    
    
    logger.info(f"Targeting Image Path: {image_path}")
    if image_path and os.path.exists(image_path):
        try:
            img_base64 = encode_image(image_path)
            
            # Dynamically determine the MIME type to avoid Google GenAI silently dropping it
            import mimetypes
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                # Fallback check
                if image_path.lower().endswith(('.jpg', '.jpeg')): mime_type = 'image/jpeg'
                elif image_path.lower().endswith('.png'): mime_type = 'image/png'
                else: mime_type = 'image/jpeg'
                
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:{mime_type};base64,{img_base64}"}
            })
            logger.info(f"Successfully attached image ({mime_type}) of size {len(img_base64)} bytes.")
        except Exception as img_err:
            logger.error(f"Failed to encode image: {img_err}")
    else:
        logger.warning(f"Image NOT ATTACHED. Path: {image_path}")
        
    messages = [("system", system_msg), HumanMessage(content=content)]

    def _execute_with_instant_fallback(msg_list):
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import ToolMessage
        import os
        
        google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # 1. Grab the tools list explicitly so it doesn't throw a NameError!
        tools_list = list(MATH_TOOLS.values())
        
        # 2. Initialize PRO
        primary_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=google_api_key,
            temperature=0.0,
            max_retries=0
        ).bind_tools(tools_list)
        
        # 3. Initialize FLASH
        fallback_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=google_api_key,
            temperature=0.0,
            max_retries=1
        ).bind_tools(tools_list)
        robust_solver = primary_llm.with_fallbacks([fallback_llm])
        tools_executed = []
        
        # Prevent state mutation loop bugs by copying original messages
        working_msgs = msg_list.copy()
        
        # 4. Invoke!
        resp = robust_solver.invoke(working_msgs)
        
        # 5. Use a WHILE loop to handle multiple tool calls gracefully with a hard cap
        iterations = 0
        while hasattr(resp, 'tool_calls') and len(resp.tool_calls) > 0 and iterations < 5:
            working_msgs.append(resp)
            for tc in resp.tool_calls:
                if tc["name"] in MATH_TOOLS:
                    try:
                        res = MATH_TOOLS[tc["name"]](**tc["args"])
                    except Exception as e:
                        res = f"Tool execution failed: {e}"
                    tools_executed.append({"tool": tc["name"], "args": tc["args"], "result": str(res)})
                    working_msgs.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))
            
            # Gemini reads the tool results and continues thinking
            resp = robust_solver.invoke(working_msgs)
            iterations += 1
            
        return resp, tools_executed

    # =================================================================
    # THIS IS THE MISSING BLOCK! It actually runs the code above!
    # =================================================================
    try:
        response, final_tools = _execute_with_instant_fallback(messages)
        
        # THE FIX: Safely extract text whether response.content is a string or a list
        raw_text = response.content
        if isinstance(raw_text, list):
            # Extract text from LangChain's list of content blocks
            text_parts = []
            for block in raw_text:
                if isinstance(block, dict) and "text" in block:
                    text_parts.append(block["text"])
                elif isinstance(block, str):
                    text_parts.append(block)
            raw_text = "".join(text_parts)
            
        if not raw_text:
            raw_text = "{}"  # Fallback if empty
            
        # Programmatically fix backslashes so JSON doesn't crash
        safe_text = str(raw_text).replace("\\", "\\\\")
        parsed = parser.parse(safe_text)
        
        parsed["tools_used_detail"] = final_tools
        parsed["status"] = "solved"
        parsed["raw_proof"] = str(raw_text)
        return parsed
    except Exception as e:
        logger.error(f"Solver Error: {e}")
        return {"status": "error", "error_message": f"Solver failed: {str(e)}"}

if __name__ == "__main__":
    print("Solver Agent with Tool Calling and Instant Fallback initialized.")