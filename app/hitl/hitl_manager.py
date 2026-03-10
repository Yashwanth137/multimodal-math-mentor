import logging

logger = logging.getLogger("HITL")

class HITLManager:
    """
    Manages coordination between automated extraction and human review.
    """
    def __init__(self, ocr_threshold: float = 0.5, asr_threshold: float = 0.8):
        self.ocr_threshold = ocr_threshold
        self.asr_threshold = asr_threshold

    def evaluate_extraction(self, extraction_result: dict, mode: str) -> dict:
        """
        Determines if an extraction result needs human intervention.
        """
        confidence = extraction_result.get("confidence", 0)
        status = extraction_result.get("status", "error")
        text = extraction_result.get("text", "")

        is_below_threshold = False
        if mode == "image":
            is_below_threshold = confidence < self.ocr_threshold
        elif mode == "audio":
            is_below_threshold = confidence < self.asr_threshold

        needs_hitl = is_below_threshold or status == "error" or not text.strip()
        
        reason = "low_confidence" if is_below_threshold else ("error" if status == "error" else "empty_input" if not text.strip() else "none")
        
        logger.info(f"HITL Eval: mode={mode}, confidence={confidence}, threshold={self.ocr_threshold if mode=='image' else self.asr_threshold}, needs_hitl={needs_hitl}, reason={reason}")
        
        return {
            "needs_hitl": needs_hitl,
            "confidence": confidence,
            "reason": reason
        }

    def evaluate_parser(self, parsed_problem: dict) -> dict:
        """
        Checks if the Parser Agent indicated a need for clarification.
        """
        needs_clarification = parsed_problem.get("needs_clarification", False)
        
        return {
            "needs_hitl": needs_clarification,
            "reason": "parser_flagged" if needs_clarification else "none"
        }

if __name__ == "__main__":
    print("HITL Manager initialized.")
