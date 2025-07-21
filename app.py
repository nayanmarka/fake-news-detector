import streamlit as st
import pickle
import os

# Utility function to load models safely
def load_model(path, model_name):
    try:
        with open(path, "rb") as file:
            model = pickle.load(file)
            return model
    except FileNotFoundError:
        st.warning(f"⚠ {model_name} model file not found.")
    except AttributeError as e:
        st.warning(f"⚠ {model_name} failed to load due to version mismatch.")
    except Exception as e:
        st.warning(f"⚠ Could not load {model_name}: {e}")
    return None

# Get current directory safely (works in Jupyter and scripts)
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

# Load vectorizer and models (NO Gradient Boosting)
vectorizer = load_model(os.path.join(current_dir, "vectorizer.pkl"), "Vectorizer")
LR = load_model(os.path.join(current_dir, "lr_model.pkl"), "Logistic Regression")
DT = load_model(os.path.join(current_dir, "dt_model.pkl"), "Decision Tree")
RF = load_model(os.path.join(current_dir, "rf_model.pkl"), "Random Forest")

# Prediction function
def manual_testing(news):
    results = {}
    if not vectorizer:
        st.error("❌ Vectorizer not loaded. Cannot proceed.")
        return results

    news_vector = vectorizer.transform([news])

    models = {
        "Logistic Regression": LR,
        "Decision Tree": DT,
        "Random Forest": RF,
    }

    for name, model in models.items():
        if model:
            try:
                pred = model.predict(news_vector)[0]
                results[name] = "FAKE" if pred == 0 else "REAL"
            except Exception as e:
                results[name] = f"❌ Failed: {e}"
        else:
            results[name] = "⚠ Not Loaded"
    return results

# Streamlit UI
st.title("📰 Fake News Detector")
st.markdown("Enter a news article below to detect whether it's *Fake* or *Real* using multiple ML models.")

news_input = st.text_area("📝 Enter News Content", height=200)

if st.button("Predict"):
    if not news_input.strip():
        st.warning("⚠ Please enter some news content.")
    else:
        predictions = manual_testing(news_input)
        if predictions:
            st.subheader("🔍 Prediction Results:")
            for model, result in predictions.items():
                st.write(f"{model}: {result}")
        else:
            st.error("❌ No predictions available.")
