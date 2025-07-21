import streamlit as st
import pickle
import os

# Safely load model files with error handling
def load_model(path, name):
    try:
        if os.path.exists(path):
            with open(path, "rb") as file:
                return pickle.load(file)
        else:
            st.warning(f"⚠️ Model file not found: {name}")
            return None
    except AttributeError:
        st.error(f"❌ Failed to load model: {name} — AttributeError (possibly corrupted or incompatible file).")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model {name}: {e}")
        return None

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load models and vectorizer
LR = load_model(os.path.join(current_dir, "lr_model.pkl"), "Logistic Regression")
DT = load_model(os.path.join(current_dir, "dt_model.pkl"), "Decision Tree")
RF = load_model(os.path.join(current_dir, "rf_model.pkl"), "Random Forest")
GB = load_model(os.path.join(current_dir, "gb_model.pkl"), "Gradient Boosting")
vectorization = load_model(os.path.join(current_dir, "vectorizer.pkl"), "TF-IDF Vectorizer")

# Manual testing function
def manual_testing(news):
    try:
        news_vector = vectorization.transform([news])
        results = {}

        if LR:
            pred_LR = LR.predict(news_vector)[0]
            results["Logistic Regression"] = "FAKE" if pred_LR == 0 else "REAL"
        if DT:
            pred_DT = DT.predict(news_vector)[0]
            results["Decision Tree"] = "FAKE" if pred_DT == 0 else "REAL"
        if RF:
            pred_RF = RF.predict(news_vector)[0]
            results["Random Forest"] = "FAKE" if pred_RF == 0 else "REAL"
        if GB:
            pred_GB = GB.predict(news_vector)[0]
            results["Gradient Boosting"] = "FAKE" if pred_GB == 0 else "REAL"

        return results

    except Exception as e:
        st.error("❌ Error during prediction.")
        st.text(str(e))
        return {}

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to detect whether it's **Fake** or **Real** using multiple ML models.")

news_input = st.text_area("Enter News Content", height=200)

if st.button("Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news content.")
    else:
        predictions = manual_testing(news_input)
        if predictions:
            st.subheader("Prediction Results:")
            for model, result in predictions.items():
                st.write(f"**{model}**: {result}")
