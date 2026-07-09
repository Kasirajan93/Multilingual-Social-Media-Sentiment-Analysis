# 🌍 Multilingual Social Media Sentiment Analysis

## Project Overview

This project analyzes sentiment from multilingual social media text using Machine Learning.

Supported Languages:
- English
- Tamil
- Hindi

The system classifies text into:
- Positive Sentiment
- Negative Sentiment

---

## Features

- Multilingual Text Support
- Text Preprocessing
- TF-IDF Feature Extraction
- Logistic Regression Classifier
- Real-Time Sentiment Prediction
- Streamlit Web Application

---

## Project Workflow

Dataset Collection
→ Data Cleaning
→ Exploratory Data Analysis
→ Feature Engineering
→ Model Training
→ Model Evaluation
→ Streamlit Deployment

---

## Models Evaluated

| Model | Accuracy |
|---------|---------|
| Logistic Regression | 79.08% |
| Naive Bayes | 77.87% |
| Linear SVM | 77.02% |

Best Model:
**Logistic Regression**

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py