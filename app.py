import streamlit as st
import pickle
import os

# Define a function to safely load models
def load_model(path, model_name):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"❌ Failed to load model: {model_name} — {e}")
        return None

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load models and vectorizer
LR = load_model(os.path.join(current_dir, "lr_model.pkl"), "Logistic Regression")
DT = load_model(os.path.join(current_dir, "dt_model.pkl"), "Decision Tree")
RF = load_model(os.path.join(current_dir, "rf_model.pkl"), "Random Forest")
GB = load_model(os.path.join(current_dir, "gb_model.pkl"), "Gradient Boosting")
vectorization = load_model(os.path.join(current_dir, "vectorizer.pkl"), "Vectorizer")

# Define the prediction function
def manual_testing(news):
    news_vector = vectorization.transform([news])
    
    results = {}
    if LR: results["Logistic Regression"] = "FAKE" if LR.predict(news_vector)[0] == 0 else "REAL"
    if DT: results["Decision Tree"] = "FAKE" if DT.predict(news_vector)[0] == 0 else "REAL"
    if RF: results["Random Forest"] = "FAKE" if RF.predict(news_vector)[0] == 0 else "REAL"
    if GB: results["Gradient Boosting"] = "FAKE" if GB.predict(news_vector)[0] == 0 else "REAL"
    
    return results

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to detect whether it's **Fake** or **Real** using multiple ML models.")

news_input = st.text_area("📝 Enter News Content", height=200)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter some news content.")
    else:
        predictions = manual_testing(news_input)
        st.subheader("🔍 Prediction Results:")
        for model, result in predictions.items():
            st.write(f"**{model}**: {result}")
