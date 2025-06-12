import streamlit as st
import joblib
import tempfile
import random
import base64
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

from scripts.extract_features import extract_features

# -------------------------------
# Load the trained KNN model
# -------------------------------
model = joblib.load("model/knn_model.pkl")

# -------------------------------
# Genre to Mood Mapping
# -------------------------------
genre_moods = {
    "blues": "🎭 Emotional",
    "classical": "🎼 Calm and Reflective",
    "country": "🌄 Heartfelt and Nostalgic",
    "disco": "🪩 Upbeat and Danceable",
    "hiphop": "🎤 Energetic and Bold",
    "jazz": "🎷 Smooth and Sophisticated",
    "metal": "🤘 Intense and Powerful",
    "pop": "🎉 Fun and Catchy",
    "rock": "🎸 Bold and Rebellious"
}

# -------------------------------
# Genre to Song Recommendations
# -------------------------------
genre_songs = {
    "blues": [
        "The Thrill Is Gone – B.B. King",
        "Me And The Devil Blues – Robert Johnson",
        "Boogie Chillen – John Lee Hooker",
        "Sweet Home Chicago – Robert Johnson",
        "Pride and Joy – Stevie Ray Vaughan"
    ],
    "classical": [
        "Canon in D – Pachelbel",
        "Clair de Lune – Debussy",
        "Für Elise – Beethoven",
        "Moonlight Sonata – Beethoven",
        "The Four Seasons – Vivaldi"
    ],
    "country": [
        "Take Me Home, Country Roads – John Denver",
        "Jolene – Dolly Parton",
        "Friends in Low Places – Garth Brooks",
        "Ring of Fire – Johnny Cash",
        "Before He Cheats – Carrie Underwood"
    ],
    "disco": [
        "Stayin’ Alive – Bee Gees",
        "I Will Survive – Gloria Gaynor",
        "Le Freak – Chic",
        "Disco Inferno – The Trammps",
        "Don’t Leave Me This Way – Thelma Houston"
    ],
    "hiphop": [
        "Juicy – The Notorious B.I.G.",
        "Lose Yourself – Eminem",
        "N.Y. State of Mind – Nas",
        "C.R.E.A.M. – Wu-Tang Clan",
        "Alright – Kendrick Lamar"
    ],
    "jazz": [
        "So What – Miles Davis",
        "Take Five – Dave Brubeck",
        "My Favorite Things – John Coltrane",
        "Round Midnight – Thelonious Monk",
        "At Last – Etta James"
    ],
    "metal": [
        "Master of Puppets – Metallica",
        "War Pigs – Black Sabbath",
        "Painkiller – Judas Priest",
        "Chop Suey! – System of a Down",
        "Holy Wars… The Punishment Due – Megadeth"
    ],
    "pop": [
        "Billie Jean – Michael Jackson",
        "Rolling in the Deep – Adele",
        "Blinding Lights – The Weeknd",
        "Like a Prayer – Madonna",
        "Shake It Off – Taylor Swift"
    ],
    "rock": [
        "Bohemian Rhapsody – Queen",
        "Stairway to Heaven – Led Zeppelin",
        "Smells Like Teen Spirit – Nirvana",
        "Sweet Child O’ Mine – Guns N’ Roses",
        "Hotel California – Eagles"
    ]
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎵 Music Genre Classifier")
st.markdown("Upload a `.wav` file to predict its genre and get recommendations!")

uploaded_file = st.file_uploader("🎼 Upload a WAV file", type="wav")

# -------------------------------
# Handle Uploaded File
# -------------------------------
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Display audio player
    st.audio(uploaded_file, format="audio/wav")

    try:
        # Feature extraction and prediction
        features = extract_features(tmp_path).reshape(1, -1)
        prediction = model.predict(features)[0]

        # Show genre and mood
        mood = genre_moods.get(prediction, "🎶 Undefined Mood")
        st.success(f"🎶 **Predicted Genre**: `{prediction}`")
        st.info(f"🧠 **Mood**: {mood}")

        # Suggested songs
        st.markdown("🎧 **Suggested Songs:**")
        suggestions = random.sample(genre_songs.get(prediction, []), 3)
        for song in suggestions:
            st.write(f"- {song}")

        # Display waveform
        st.markdown("📊 **Waveform**")
        y, sr = librosa.load(tmp_path)
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Audio Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ Error: {e}")
