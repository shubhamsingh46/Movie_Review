import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬", layout="centered")

# Custom CSS for UI
st.markdown("""
    <style>
    body {
        background-color: #0f172a;
    }
    .main {
        background-color: #0f172a;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #38bdf8;
    }
    .subtitle {
        text-align: center;
        color: #cbd5f5;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .stTextArea textarea {
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        background-color: #38bdf8;
        color: white;
        font-size: 18px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze whether a movie review is Positive or Negative</div>', unsafe_allow_html=True)

# Input Box
review = st.text_area("✍️ Enter your review below:", height=150, placeholder="Type your movie review here...")

# Prediction Button
if st.button("🔍 Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        data = vectorizer.transform([review]).toarray()
        prediction = model.predict(data)

        st.markdown("---")

        if prediction[0] == 1:
            st.success("✅ Positive Review 😊")
            st.balloons()
        else:
            st.error("❌ Negative Review 😡")

# Footer
st.markdown("---")
st.markdown(
    "<center><small>Built with ❤️ using Streamlit | Practical 8 Project</small></center>",
    unsafe_allow_html=True
)