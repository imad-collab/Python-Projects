# models/model_loader.py

from transformers import pipeline

# Load HuggingFace transformers
def load_detection_model():
    return pipeline("text-classification")
