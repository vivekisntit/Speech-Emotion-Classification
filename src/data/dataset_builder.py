import os
import pandas as pd

from src.config.config import DATA_RAW


RAVDESS_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

CREMA_MAP = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fearful",
    "DIS": "disgust"
}

SAVEE_MAP = {
    "a": "angry",
    "d": "disgust",
    "f": "fearful",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprised"
}


def parse_ravdess(file):
    parts = file.split("-")
    emotion_code = parts[2]
    return RAVDESS_MAP.get(emotion_code)


def parse_tess(file):
    emotion = file.split("_")[-1].replace(".wav", "").lower()

    if emotion == "fear":
        return "fearful"

    if emotion == "ps":
        return "surprised"

    return emotion


def parse_crema(file):
    parts = file.split("_")
    emotion_code = parts[2]
    return CREMA_MAP.get(emotion_code)


def parse_savee(file):
    code = file.split("_")[1]

    if code.startswith("sa"):
        return "sad"

    if code.startswith("su"):
        return "surprised"

    return SAVEE_MAP.get(code[0])


def build_metadata():

    rows = []

    for root, dirs, files in os.walk(DATA_RAW):

        for file in files:

            if not file.endswith(".wav"):
                continue

            filepath = os.path.join(root, file)

            if "ravdess" in root.lower():

                emotion = parse_ravdess(file)

            elif "tess" in root.lower():

                emotion = parse_tess(file)

            elif "crema" in root.lower():

                emotion = parse_crema(file)

            elif "savee" in root.lower():

                emotion = parse_savee(file)

            else:
                continue

            if emotion is None:
                continue

            rows.append({
                "filepath": filepath,
                "emotion": emotion
            })

    df = pd.DataFrame(rows)

    df.to_csv("data/metadata.csv", index=False)

    print("Metadata created:", df.shape)


if __name__ == "__main__":
    build_metadata()