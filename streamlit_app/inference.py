import os
import torch
import torch.nn.functional as F
import re
import streamlit as st  # Required for caching
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------
# CONFIG
# ----------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "harneetkaur09/mentalbert-mental-health"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal"
]

label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

# ----------------------
# LOAD MODEL (CACHED)
# ----------------------
@st.cache_resource
def load_model():
    """
    Loads the tokenizer and model.
    Cached by Streamlit so it doesn't reload on every interaction.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=7,
        id2label=id2label,
        label2id=label2id,
        token=HF_TOKEN
    )

    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# Initialize the model and tokenizer globally
tokenizer, model = load_model()

# ----------------------
# PREPROCESS
# ----------------------
def minimal_clean(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|@\w+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------
# PREDICTION FUNCTION
# ----------------------
def predict_text(text):
    text = minimal_clean(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze()

    results = {
        id2label[i]: float(probs[i])
        for i in range(len(probs))
    }
    
    # Get the label with the highest score
    predicted_label = max(results, key=results.get)

    explanation = None  # Placeholder for future interpretability (LIME/SHAP)
    
    return predicted_label, results, explanation
