import os
import pandas as pd
import librosa
import numpy as np
import joblib

from src.config.config import DATA_PROCESSED


EMOTION_LABELS = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7
}


def extract_features():
    df = pd.read_csv("data/metadata.csv")
    X = []
    y = []
    for idx, row in df.iterrows():
        path = row["filepath"]
        emotion = row["emotion"]
        try:
            signal, sr = librosa.load(path, res_type="kaiser_fast")
            mfcc = np.mean(
                librosa.feature.mfcc(
                    y=signal,
                    sr=sr,
                    n_mfcc=40
                ).T,
                axis=0
            )
            X.append(mfcc)
            y.append(EMOTION_LABELS[emotion])
        except Exception as e:
            print("Error loading:", path)
            print(e)
    X = np.array(X)
    y = np.array(y)

    os.makedirs(DATA_PROCESSED, exist_ok=True)
    joblib.dump(X, os.path.join(DATA_PROCESSED, "X.joblib"))
    joblib.dump(y, os.path.join(DATA_PROCESSED, "y.joblib"))

    print("Features saved:", X.shape)

if __name__ == "__main__":
    extract_features()