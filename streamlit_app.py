import streamlit as st
import joblib
import tempfile
from scripts.extract_features import extract_features

model = joblib.load("model/knn_model.pkl")

st.title("🎵 Music Genre Classifier")
st.markdown("Upload a .wav file to predict the genre.")

uploaded_file = st.file_uploader("Upload .wav file", type="wav")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        features = extract_features(tmp_path).reshape(1, -1)
        prediction = model.predict(features)[0]
        st.success(f"🎶 Predicted Genre: **{prediction}**")
    except Exception as e:
        st.warning(f"⚠️ Error: {e}")
