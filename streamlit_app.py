import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import joblib
import random
from scripts.extract_features import extract_features

# Load model
model = joblib.load("model/knn_model.pkl")

genre_moods = {
    "blues": "🎭 Emotional", "classical": "🎼 Calm and Reflective", "country": "🌄 Heartfelt and Nostalgic",
    "disco": "🪩 Upbeat and Danceable", "hiphop": "🎤 Energetic and Bold", "jazz": "🎷 Smooth and Sophisticated",
    "metal": "🤘 Intense and Powerful", "pop": "🎉 Fun and Catchy", "rock": "🎸 Bold and Rebellious"
}

genre_songs = {
    "blues": ["The Thrill Is Gone – B.B. King", "Sweet Home Chicago – Robert Johnson"],
    "classical": ["Canon in D – Pachelbel", "Clair de Lune – Debussy"],
    "country": ["Take Me Home, Country Roads – John Denver", "Jolene – Dolly Parton"],
    "disco": ["Stayin’ Alive – Bee Gees", "I Will Survive – Gloria Gaynor"],
    "hiphop": ["Juicy – The Notorious B.I.G.", "Lose Yourself – Eminem"],
    "jazz": ["So What – Miles Davis", "Take Five – Dave Brubeck"],
    "metal": ["Master of Puppets – Metallica", "War Pigs – Black Sabbath"],
    "pop": ["Billie Jean – Michael Jackson", "Rolling in the Deep – Adele"],
    "rock": ["Bohemian Rhapsody – Queen", "Stairway to Heaven – Led Zeppelin"]
}

# Waveform plot function
def plot_waveform(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("📈 Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Waveform Error: {e}")

# Streamlit UI
st.title("🎵 Music Genre Classifier")
uploaded_file = st.file_uploader("Upload a .wav file", type="wav")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.subheader("📈 Audio Waveform")
    plot_waveform(tmp_path)

    try:
        features = extract_features(tmp_path).reshape(1, -1)
        prediction = model.predict(features)[0]
        mood = genre_moods.get(prediction, "🎶 Undefined Mood")
        st.success(f"🎶 Predicted Genre: **{prediction}**\n\n🧠 Mood: *{mood}*")

        st.subheader("🎧 Suggested Songs")
        for song in random.sample(genre_songs.get(prediction, []), 2):
            st.markdown(f"- {song}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
