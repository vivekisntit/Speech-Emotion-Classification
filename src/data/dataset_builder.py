import os
import pandas as pd

from src.config.config import DATA_RAW

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def build_metadata():

    rows = []

    for root, dirs, files in os.walk(DATA_RAW):

        for file in files:

            if not file.endswith(".wav"):
                continue

            filepath = os.path.join(root, file)

            # RAVDESS files follow numeric naming
            if "-" in file:

                parts = file.split("-")

                emotion_code = parts[2]

                emotion = EMOTION_MAP.get(emotion_code, "unknown")

            else:
                # TESS naming example: OAF_angry.wav
                name = file.lower()

                if "angry" in name:
                    emotion = "angry"
                elif "happy" in name:
                    emotion = "happy"
                elif "sad" in name:
                    emotion = "sad"
                elif "fear" in name:
                    emotion = "fearful"
                elif "disgust" in name:
                    emotion = "disgust"
                elif "ps" in name or "surprise" in name:
                    emotion = "surprised"
                else:
                    emotion = "neutral"

            rows.append({
                "filepath": filepath,
                "emotion": emotion
            })

    df = pd.DataFrame(rows)

    os.makedirs("data", exist_ok=True)

    df.to_csv("data/metadata.csv", index=False)

    print("Metadata created:", len(df), "files")


if __name__ == "__main__":
    build_metadata()