import streamlit as st
import pickle
from preprocess import clean_text

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# UI Title
st.title("⭐ Product Review Rating Predictor")

st.write("Enter a product review to predict its rating")

# Input box
review = st.text_area("📝 Enter your review:")

if st.button("Predict"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review")
    else:
        # Preprocess
        cleaned = clean_text(review)
        cleaned_text = " ".join(cleaned)

        # Transform
        vector = vectorizer.transform([cleaned_text])

        # Predict
        rating = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0]
        confidence = max(proba)

        # Output
        st.subheader("🔍 Prediction Result")

        st.write(f"⭐ Predicted Rating: {rating}/2")

        if rating == 2:
            st.success("😊 Positive Review")
        elif rating == 1:
            st.info("😐 Neutral Review")
        else:
            st.error("😞 Negative Review")

        st.write(f"📊 Confidence: {confidence:.2%}")