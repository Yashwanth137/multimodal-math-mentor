import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelFactory")

def get_supervisor_model() -> BaseChatModel:
    """
    Returns the high-speed Supervisor node (Groq - Llama 3.3 70B).
    Used for triage, RAG, and final formatting.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment. Supervisor node requires Groq.")
    
    model_name = os.getenv("SUPERVISOR_MODEL", "llama-3.3-70b-versatile")
    logger.info(f"Initializing Supervisor Node: {model_name} (Groq)")
    
    return ChatGroq(
        model=model_name,
        groq_api_key=groq_api_key,
        temperature=0.1,
        max_retries=2,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

def get_solver_model(model_name: str = None) -> BaseChatModel:
    """
    Returns the reasoning/vision Solver node (Google AI Studio - Gemini).
    Implements model fallback: gemini-2.5-flash -> gemini-2.0-flash.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Solver node requires Gemini.")
    
    # Updated model list based on current audit requirements
    primary_model = os.getenv("SOLVER_MODEL", "gemini-2.5-flash")
    fallback_model = "gemini-2.0-flash"
    
    selected_model = model_name or primary_model
    
    # Robust fallback logic
    models_to_try = [selected_model, fallback_model] if selected_model != fallback_model else [fallback_model]
    
    last_err = None
    for m in models_to_try:
        try:
            # Ensure proper prefixing for AI Studio routing
            full_model_name = m if m.startswith("models/") else f"models/{m}"
            logger.info(f"Attempting to initialize Solver Node: {full_model_name}")
            
            return ChatGoogleGenerativeAI(
                model=full_model_name,
                google_api_key=google_api_key,
                temperature=0.0,
                max_retries=2,
                request_timeout=60
            )
        except Exception as e:
            last_err = e
            logger.warning(f"Failed to initialize model {m}. Trying fallback if available.")
            continue
            
    raise RuntimeError(f"All solver models failed to initialize. Last error: {last_err}")

# Keeping get_model for backward compatibility during refactor, but it will be phased out
def get_model(temperature: float = 0) -> BaseChatModel:
    """Legacy get_model - now routes to Supervisor by default."""
    return get_supervisor_model()

if __name__ == "__main__":
    try:
        sup = get_supervisor_model()
        sol = get_solver_model()
        print("Aethyln Labs Dual-Node Model Factory initialized.")
    except Exception as e:
        print("Error initializing Aethyln Labs Dual-Node Model Factory.")
