import pytesseract
from PIL import Image
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OCR")

# Configure Tesseract path for Windows if provided
tesseract_path = os.getenv("TESSERACT_PATH")
if tesseract_path:
    logger.info(f"Setting Tesseract path to: {tesseract_path}")
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

def run_ocr(image_path: str) -> dict:
    """
    Performs OCR on the given image and returns extracted text and confidence.
    """
    try:
        logger.info(f"Running OCR on: {image_path}")
        img = Image.open(image_path)
        # Using pytesseract to get both text and data (for confidence)
        text = pytesseract.image_to_string(img)
        
        # Get detailed data to calculate average confidence
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data['conf'] if int(c) != -1]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        confidence_norm = avg_confidence / 100.0
        
        return {
            "text": text.strip(),
            "confidence": confidence_norm,
            "status": "success"
        }
    except Exception as e:
        logger.error("OCR Error occurred.")
        return {
            "text": "",
            "confidence": 0,
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    # Simple test check
    print("OCR Pipeline initialized.")
