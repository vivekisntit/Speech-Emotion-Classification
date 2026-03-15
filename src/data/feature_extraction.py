import os
import librosa
import numpy as np
import joblib

from src.config.config import DATA_RAW, DATA_PROCESSED

def extract_features():

    data = []

    for subdir, dirs, files in os.walk(DATA_RAW):

        for file in files:

            if file.endswith(".wav"):

                try:
                    path = os.path.join(subdir, file)

                    signal, sr = librosa.load(path, res_type="kaiser_fast")

                    mfcc = np.mean(
                        librosa.feature.mfcc(
                            y=signal,
                            sr=sr,
                            n_mfcc=40
                        ).T,
                        axis=0
                    )

                    label = int(file[7:8]) - 1

                    data.append([mfcc, label])

                except:
                    continue

    X, y = zip(*data)

    X = np.array(X)
    y = np.array(y)

    os.makedirs(DATA_PROCESSED, exist_ok=True)

    joblib.dump(X, os.path.join(DATA_PROCESSED, "X.joblib"))
    joblib.dump(y, os.path.join(DATA_PROCESSED, "y.joblib"))

    print("Features saved")