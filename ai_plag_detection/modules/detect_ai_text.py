# modules/detect_ai_text.py

from models.model_loader import load_detection_model

# Function to detect AI-generated text
def detect_ai_text(text):
    detection_model = load_detection_model()
    result = detection_model(text)

    if result:
        return {
            "label": result[0]['label'],
            "confidence": result[0]['score']
        }
    return {"message": "AI detection failed"}
