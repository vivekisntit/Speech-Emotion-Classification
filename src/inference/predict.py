import librosa
import numpy as np
from keras.models import load_model

from src.config.config import MODEL_DIR

EMOTIONS = [
"neutral",
"calm",
"happy",
"sad",
"angry",
"fearful",
"disgust",
"surprised"
]

def predict(file):

    model = load_model(MODEL_DIR + "/emotion_model.h5")

    data, sr = librosa.load(file)

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=data,
            sr=sr,
            n_mfcc=40
        ).T,
        axis=0
    )

    x = np.expand_dims(mfcc, axis=0)
    x = np.expand_dims(x, axis=2)

    pred = model.predict(x)
    label = EMOTIONS[np.argmax(pred)]

    print("Prediction:", label)