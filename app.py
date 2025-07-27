import streamlit as st
import pickle
import requests

# ✅ Securely get GNews API key from Streamlit Secrets
api_key = st.secrets.get("GNEWS_API_KEY")

if not api_key:
    st.error("🚨 GNEWS_API_KEY is missing from Streamlit secrets!")
    st.stop()

# ✅ Load vectorizer and model
try:
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    lr_model = pickle.load(open("lr_model.pkl", "rb"))  # Make sure filename matches
except FileNotFoundError as e:
    st.error(f"❌ Model or vectorizer file not found: {e}")
    st.stop()

# ✅ UI Title
st.title("📰 Fake News Detection using ML")
st.markdown("Detect whether news is **Real** or **Fake**, including live headlines!")

# ✅ Predict function
def predict_news(text):
    transformed_text = vectorizer.transform([text])
    prediction = lr_model.predict(transformed_text)
    return "Real News" if prediction[0] == 0 else "Fake News"

# ✅ Fetch live headlines using GNews
def fetch_top_news():
    url = f"https://gnews.io/api/v4/top-headlines?lang=en&country=in&max=5&token={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        return data.get("articles", [])
    except Exception as e:
        st.error(f"❌ Failed to fetch news: {e}")
        return []

# ✅ Manual entry section
st.subheader("✍️ Enter Custom News for Detection")
user_input = st.text_area("Enter news content here")

if st.button("Check News"):
    if user_input.strip():
        result = predict_news(user_input)
        st.success(f"✅ Prediction: {result}")
    else:
        st.warning("⚠️ Please enter some news content.")

# ✅ Live headlines section
st.markdown("---")
st.subheader("🌐 Detect Fake News from Live Headlines")

if st.button("Fetch & Predict Top News"):
    articles = fetch_top_news()
    if not articles:
        st.warning("⚠️ No news articles fetched.")
    else:
        for i, article in enumerate(articles, 1):
            title = article.get("title", "No Title")
            description = article.get("description", "")
            full_text = f"{title} {description}"

            prediction = predict_news(full_text)

            st.markdown(f"**{i}. {title}**")
            st.markdown(f"*Prediction:* `{prediction}`")
            st.markdown("---")
