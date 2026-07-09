import streamlit as st
import joblib
import re

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Multilingual Sentiment Analysis",
    page_icon="🌍",
    layout="centered"
)

# --------------------------------------------------
# Load Model and Vectorizer
# --------------------------------------------------
model = joblib.load("models/logistic_regression_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# --------------------------------------------------
# Text Cleaning Function
# --------------------------------------------------
def clean_text(text):

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.replace("#", "")
    text = re.sub(r"\d+", "", text)

    text = re.sub(
        r"[^\w\s\u0B80-\u0BFF\u0900-\u097F]",
        "",
        text
    )

    text = text.lower()
    text = " ".join(text.split())

    return text


# --------------------------------------------------
# App Header
# --------------------------------------------------
st.title("🌍 Multilingual Social Media Sentiment Analysis")

st.markdown("""
Analyze sentiment from **English, Tamil, and Hindi**
social media text using **Machine Learning (TF-IDF + Logistic Regression)**.
""")

st.markdown("---")

# --------------------------------------------------
# User Input
# --------------------------------------------------
user_text = st.text_area(
    "Enter Social Media Text",
    height=150,
    placeholder="Type your text here..."
)

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
if st.button("Predict Sentiment"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Clean Text
        cleaned = clean_text(user_text)

        # Vectorize
        vectorized = vectorizer.transform([cleaned])

        # Predict
        prediction = model.predict(vectorized)[0]

        # Confidence Score
        confidence = model.predict_proba(vectorized).max() * 100

        st.markdown("## Prediction Result")

        if prediction == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")

        # Confidence Score Card
        st.metric(
            label="Confidence Score",
            value=f"{confidence:.2f}%"
        )

        # Cleaned Text Display
        st.markdown("### Cleaned Text")

        st.code(cleaned)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")

st.caption(
    "Developed by Kasi Rajan | AI/ML Internship Project"
)