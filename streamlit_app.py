import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import tempfile

# Load the trained CNN model
model = tf.keras.models.load_model('cnn_genre_model.h5')

# Genre list
genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'rock']

# Suggested songs dictionary
genre_songs = {
    'blues': ['The Thrill Is Gone - B.B. King', 'Pride and Joy - Stevie Ray Vaughan', 'Sweet Home Chicago - Robert Johnson'],
    'classical': ['Moonlight Sonata - Beethoven', 'Four Seasons: Spring - Vivaldi', 'Clair de Lune - Debussy'],
    'country': ['Take Me Home, Country Roads - Lana Del Rey', 'Before He Cheats - Carrie Underwood', 'Folsom Prison Blues - Johnny Cash'],
    'disco': ['Stayin’ Alive - Bee Gees', 'I Will Survive - Gloria Gaynor', 'Le Freak - Chic'],
    'hiphop': ['Lose Yourself - Eminem', 'The Message - Grandmaster Flash', 'God\'s Plan - Drake'],
    'jazz': ['So What - Miles Davis', 'Take the A Train - Duke Ellington', 'Fly Me to the Moon - Frank Sinatra'],
    'metal': ['Master of Puppets - Metallica', 'Iron Man - Black Sabbath', 'Chop Suey! - System of a Down'],
    'pop': ['Blinding Lights - The Weeknd', 'Levitating - Dua Lipa', 'Flowers - Miley Cyrus'],
    'rock': ['Bohemian Rhapsody - Queen', 'Stairway to Heaven - Led Zeppelin', 'Hotel California - Eagles']
}

# Extract mel-spectrogram
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)
    S_DB = librosa.util.fix_length(S_DB, size=660, axis=1)  # pad or trim to fixed length
    S_DB = np.expand_dims(S_DB, axis=-1)  # add channel dimension
    return S_DB

# Show spectrogram
def show_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title('Mel Spectrogram')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    st.pyplot(fig)

# Streamlit UI
st.set_page_config(page_title="🎵 Music Genre Classifier", layout="centered")
st.title("🎵 Music Genre Classifier")
st.write("Upload a `.wav` file to predict its music genre.")

uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Show spectrogram
    st.subheader("📊 Spectrogram")
    show_spectrogram(file_path)

    # Extract features and predict
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)  # batch dimension
    prediction = model.predict(features)[0]
    predicted_index = np.argmax(prediction)
    predicted_genre = genres[predicted_index]

    st.subheader("🎼 Predicted Genre:")
    st.success(predicted_genre.capitalize())

    # Show suggestions
    st.subheader("🎧 Suggested Songs:")
    suggestions = genre_songs.get(predicted_genre, [])
    for song in suggestions:
        st.markdown(f"- {song}")
