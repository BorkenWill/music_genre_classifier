# Try it here!
https://musicgenreclassifier-vgsdtdwfgus3izudfbknm5.streamlit.app/

# 🎵 Music Genre Classification using Audio Features & KNN

This project classifies music clips into genres (e.g., rock, classical, jazz, etc.) by extracting audio features from .wav files and using a K-Nearest Neighbors (KNN) classifier.

# 📂 Dataset
Based on the GTZAN Dataset

Contains genres: blues, classical, country, disco, hiphop, jazz, metal, pop, rock

# 🧠 Features Extracted
MFCCs (Mel Frequency Cepstral Coefficients)

Spectral Centroid

Zero-Crossing Rate

Optionally: RMS Energy, Chroma, Rolloff, etc.

# 🛠️ Model
Trained a KNN (K-Nearest Neighbors) model using extracted features

Used scikit-learn and joblib to save/load the model

# 🚀 Streamlit Web App
Upload a .wav music file

The app extracts features and predicts its genre using the trained model
