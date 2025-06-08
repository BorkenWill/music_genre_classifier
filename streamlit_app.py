import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile

from your_module import extract_features  # or define extract_features here

# Load model
model = joblib.load("model/knn_model.pkl")

st.title("🎵 Music Genre Classifier")
st.markdown("Upload a music file (.wav) and classify its genre.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file:
    try:
        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Extract features
        features = extract_features(tmp_path)
        features = features.reshape(1, -1)

        # Predict
        prediction = model.predict(features)[0]
        st.success(f"🎧 Predicted Genre: **{prediction}**")
        
    except Exception as e:
        st.warning(f"⚠️ Error processing file: {e}")
