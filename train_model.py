import os
import pandas as pd
from extract_features import extract_features
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load files and extract features
genres = ['rock', 'classical', 'pop']
data = []

for genre in genres:
    folder = f"audio/{genre}"
    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            path = os.path.join(folder, filename)
            features = extract_features(path)
            features['genre'] = genre
            data.append(features)

# Create DataFrame and train model
df = pd.DataFrame(data)
X = df.drop('genre', axis=1)
y = df['genre']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)
with open("model/knn_model.pkl", "wb") as f:
    pickle.dump(model, f)
