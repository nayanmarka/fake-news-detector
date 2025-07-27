import streamlit as st
import pickle
import numpy as np
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file (for GNews API Key)
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("GNEWS_API_KEY")

# Load vectorizer and model
try:
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    lr_model = pickle.load(open("Lr model.pkl", "rb"))  # Your Logistic Regression model
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# Title
st.title("📰 Fake News Detection using ML")
st.markdown("Get real-time news and detect if it's **Fake** or **Real**.")

# Function to fetch top news headlines
def fetch_top_news():
    url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=5&token={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        return data.get("articles", [])
    except Exception as e:
        st.error(f"❌ Error fetching news: {e}")
        return []

# Function to predict fake/real
def predict_news(text):
    transformed_text = vectorizer.transform([text])
    prediction = lr_model.predict(transformed_text)
    return "Fake News" if prediction[0] == 0 else "Real News"

# Option to enter custom news
st.subheader("✍️ Enter Custom News for Detection")
user_input = st.text_area("Enter news content here")

if st.button("Check News"):
    if user_input.strip():
        result = predict_news(user_input)
        st.success(f"✅ Prediction: {result}")
    else:
        st.warning("⚠️ Please enter some news content.")

# Divider
st.markdown("---")

# Option to test real-time GNews headlines
st.subheader("🌐 Detect Fake News from Live Headlines")
if st.button("Fetch & Predict Top News"):
    articles = fetch_top_news()
    if not articles:
        st.warning("No news articles found.")
    else:
        for i, article in enumerate(articles, start=1):
            title = article.get("title", "No Title")
            description = article.get("description", "")
            combined = title + " " + description

            prediction = predict_news(combined)

            st.markdown(f"**{i}. {title}**")
            st.markdown(f"*Prediction:* `{prediction}`")
            st.markdown("---")
