import streamlit as st
import numpy as np
import librosa
import joblib
from sklearn.pipeline import Pipeline  # Ensure Pipeline is imported for model loading

# Load the trained model
data = joblib.load("model/knn_model.pkl")
model = data["model"]  # ✅ Extract actual model
# Optional: label_encoder = data["label_encoder"] if included

# Feature extraction function
def extract_features(file):
    y, sr = librosa.load(file, duration=30)
    
    # Extract spectral features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)

    features = np.hstack([zcr, centroid, mfcc])
    return features

# Streamlit UI
st.title("🎵 Music Genre Classification")

st.write("Upload a 30-second audio file to predict its genre:")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    try:
        with st.spinner("Extracting features and predicting..."):
            features = extract_features(uploaded_file)
            prediction = model.predict([features])[0]
            st.success(f"🎧 Predicted Genre: **{prediction}**")
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
