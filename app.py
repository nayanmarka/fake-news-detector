import streamlit as st
import pickle
import os

# Function to safely load models
def load_model(path, model_name):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.warning(f"⚠️ Could not load {model_name}: {e}")
        return None

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load vectorizer and models
vectorizer = load_model(os.path.join(current_dir, "vectorizer.pkl"), "Vectorizer")
LR = load_model(os.path.join(current_dir, "lr_model.pkl"), "Logistic Regression")
DT = load_model(os.path.join(current_dir, "dt_model.pkl"), "Decision Tree")
RF = load_model(os.path.join(current_dir, "rf_model.pkl"), "Random Forest")

# Only try loading Gradient Boosting if you're sure it's compatible
GB = load_model(os.path.join(current_dir, "gb_model.pkl"), "Gradient Boosting")

# Manual prediction function
def manual_testing(news):
    results = {}
    try:
        news_vector = vectorizer.transform([news])
        if LR:
            pred = LR.predict(news_vector)[0]
            results["Logistic Regression"] = "FAKE" if pred == 0 else "REAL"
        if DT:
            pred = DT.predict(news_vector)[0]
            results["Decision Tree"] = "FAKE" if pred == 0 else "REAL"
        if RF:
            pred = RF.predict(news_vector)[0]
            results["Random Forest"] = "FAKE" if pred == 0 else "REAL"
        if GB:
            pred = GB.predict(news_vector)[0]
            results["Gradient Boosting"] = "FAKE" if pred == 0 else "REAL"
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
    return results

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to detect whether it's **Fake** or **Real** using multiple ML models.")

news_input = st.text_area("📝 Enter News Content", height=200)

if st.button("Predict"):
    if not news_input.strip():
        st.warning("⚠️ Please enter some news content.")
    else:
        predictions = manual_testing(news_input)
        if predictions:
            st.subheader("🔍 Prediction Results:")
            for model, result in predictions.items():
                st.write(f"**{model}**: {result}")
        else:
            st.error("❌ No predictions available. All models might have failed.")
