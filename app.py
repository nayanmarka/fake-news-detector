import streamlit as st
import pickle
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load models and vectorizer with absolute paths
LR = pickle.load(open(os.path.join(current_dir, "lr_model.pkl"), "rb"))
DT = pickle.load(open(os.path.join(current_dir, "dt_model.pkl"), "rb"))
RF = pickle.load(open(os.path.join(current_dir, "rf_model.pkl"), "rb"))
GB = pickle.load(open(os.path.join(current_dir, "gb_model.pkl"), "rb"))
vectorization = pickle.load(open(os.path.join(current_dir, "vectorizer.pkl"), "rb"))

# Manual prediction function
def manual_testing(news):
    news_vector = vectorization.transform([news])
    pred_LR = LR.predict(news_vector)[0]
    pred_DT = DT.predict(news_vector)[0]
    pred_RF = RF.predict(news_vector)[0]
    pred_GB = GB.predict(news_vector)[0]

    results = {
        "Logistic Regression": "FAKE" if pred_LR == 0 else "REAL",
        "Decision Tree": "FAKE" if pred_DT == 0 else "REAL",
        "Random Forest": "FAKE" if pred_RF == 0 else "REAL",
        "Gradient Boosting": "FAKE" if pred_GB == 0 else "REAL",
    }
    return results

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to detect whether it's **Fake** or **Real** using multiple ML models.")

news_input = st.text_area("Enter News Content", height=200)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        predictions = manual_testing(news_input)
        st.subheader("Prediction Results:")
        for model, result in predictions.items():
            st.write(f"**{model}**: {result}")
