import streamlit as st
import joblib
import tempfile
from scripts.extract_features import extract_features

# Load the model
model = joblib.load("model/knn_model.pkl")

# Page config
st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 600px;
        margin: auto;
        padding-top: 3rem;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #333;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .genre-box {
        margin-top: 2rem;
        padding: 1.5rem;
        background-color: #e3f2fd;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        color: #1e88e5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# App UI
st.markdown('<div class="title">🎵 Music Genre Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a WAV file to predict its music genre</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("🎧 Upload your .wav file", type="wav")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path).reshape(1, -1)
        prediction = model.predict(features)[0]
        st.markdown(f'<div class="genre-box">🎶 Predicted Genre: <br><span>{prediction}</span></div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"⚠️ Error: {e}")

