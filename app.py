import streamlit as st
import pickle
import re
import string

# Load models and vectorizer
LR = pickle.load(open("lr_model.pkl", "rb"))
DT = pickle.load(open("dt_model.pkl", "rb"))
GB = pickle.load(open("gb_model.pkl", "rb"))
RF = pickle.load(open("rf_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def output_label(n):
    return "Fake News" if n == 0 else "Not A Fake News"

# Streamlit UI
st.title("📰 Fake News Detector")

news = st.text_area("Enter the news text here")

if st.button("Predict"):
    processed_news = wordopt(news)
    vect_text = vectorizer.transform([processed_news])

    pred_LR = output_label(LR.predict(vect_text)[0])
    pred_DT = output_label(DT.predict(vect_text)[0])
    pred_GB = output_label(GB.predict(vect_text)[0])
    pred_RF = output_label(RF.predict(vect_text)[0])

    st.subheader("Model Predictions:")
    st.write(f"**Logistic Regression**: {pred_LR}")
    st.write(f"**Decision Tree**: {pred_DT}")
    st.write(f"**Gradient Boosting**: {pred_GB}")
    st.write(f"**Random Forest**: {pred_RF}")
# update for github pus
