import whisper
import os
import torch
from dotenv import load_dotenv

load_dotenv()

# Global model instance for efficiency (lazy loading)
_whisper_model = None

def get_whisper_model(model_name: str = "base"):
    global _whisper_model
    if _whisper_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _whisper_model = whisper.load_model(model_name, device=device)
    return _whisper_model

def run_asr(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribes audio to text using OpenAI Whisper.
    """
    try:
        model = get_whisper_model(model_size)
        result = model.transcribe(audio_path)
        
        # Calculate segments confidence (averaging)
        segments = result.get("segments", [])
        avg_confidence = 0
        if segments:
            avg_confidence = sum([s.get("avg_logprob", 0) for s in segments]) / len(segments)
            # Rough conversion of logprob to 0-1 confidence
            import math
            avg_confidence = math.exp(avg_confidence)
            
        return {
            "text": result["text"].strip(),
            "confidence": avg_confidence,
            "status": "success"
        }
    except Exception as e:
        return {
            "text": "",
            "confidence": 0,
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    print("ASR Pipeline initialized.")
