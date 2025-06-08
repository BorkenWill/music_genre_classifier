import os
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from scripts.extract_features import extract_features

DATASET_PATH = "dataset"
GENRES = os.listdir(DATASET_PATH)

X, y = [], []

for genre in GENRES:
    genre_folder = os.path.join(DATASET_PATH, genre)
    for file in os.listdir(genre_folder):
        if file.endswith(".wav"):
            path = os.path.join(genre_folder, file)
            try:
                features = extract_features(path)
                X.append(features)
                y.append(genre)
            except Exception as e:
                print(f"Error with {file}: {e}")

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "model/knn_model.pkl")
print("✅ Model saved.")
