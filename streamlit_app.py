import streamlit as st
import joblib
import tempfile
from scripts.extract_features import extract_features

# Load the model
model = joblib.load("model/knn_model.pkl")

# Genre to mood mapping
genre_moods = {
    "blues": "🎷 Emotional / Reflective",
    "classical": "🎻 Calm / Elegant",
    "country": "🤠 Nostalgic / Heartfelt",
    "disco": "🪩 Energetic / Fun",
    "hiphop": "🎤 Confident / Rhythmic",
    "jazz": "🎺 Smooth / Sophisticated",
    "metal": "🤘 Intense / Aggressive",
    "pop": "🎧 Upbeat / Catchy",
    "rock": "🎸 Powerful / Rebellious",
    "reggae": "🌴 Relaxed / Groovy"
}

# Page config
st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")

# Custom CSS
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
        font-size: 1.2rem;
        font-weight: bold;
        color: #1e88e5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .mood {
        font-size: 1rem;
        font-weight: normal;
        color: #555;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🎵 Music Genre Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a WAV file to predict its genre and mood</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("🎧 Upload your .wav file", type="wav")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path).reshape(1, -1)
        prediction = model.predict(features)[0]
        mood = genre_moods.get(prediction.lower(), "🎼 Unknown mood")

        st.markdown(f'''
            <div class="genre-box">
                🎶 <strong>Predicted Genre:</strong> {prediction}<br>
                <div class="mood">🧠 <strong>Mood:</strong> {mood}</div>
            </div>
        ''', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"⚠️ Error: {e}")
