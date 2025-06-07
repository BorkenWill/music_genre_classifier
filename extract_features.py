import librosa
import numpy as np

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=10)
    features = {
        'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
        'centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
    }
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}'] = np.mean(mfcc[i])
    return features
