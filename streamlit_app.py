import streamlit as st
import librosa
import numpy as np
import pickle
import os

from extract_features import extract_features

# Load trained model
with open("model/knn_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🎵 Music Genre Classifier")

uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features
    features = extract_features("temp.wav")
    features_array = np.array([list(features.values())])
    
    # Predict
    prediction = model.predict(features_array)[0]
    st.success(f"🎧 Predicted Genre: **{prediction.capitalize()}**")

    # Optional: Visualize features
    st.subheader("Extracted Features")
    st.write({k: round(v, 2) for k, v in features.items()})
