import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np

# ===== File Paths =====
MODEL_PATH = "fine-tuned-airline-model-1000"  # matches your folder
LABEL_ENCODER_PATH = "label_encoder.pkl"     # matches your file

# ===== Load model, tokenizer, and label encoder =====
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return tokenizer, model, label_encoder

tokenizer, model, label_encoder = load_model()
model.eval()

# ===== Prediction function =====
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
    predicted_class = np.argmax(probs)
    label = label_encoder.inverse_transform([predicted_class])[0]
    return label, probs

# ===== Streamlit UI =====
st.title("✈ Airline Tweet Sentiment Classifier")
st.write("Enter a tweet to classify it as **positive**, **neutral**, or **negative**.")

user_input = st.text_area("Tweet text:", "")

if st.button("Classify"):
    if user_input.strip():
        label, probs = predict_sentiment(user_input)
        st.subheader(f"Prediction: {label}")
        st.write("Confidence scores:")
        for i, class_label in enumerate(label_encoder.classes_):
            st.write(f"{class_label}: {probs[i]:.4f}")
    else:
        st.warning("Please enter some text to classify.")

