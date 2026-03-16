import os
import json
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config.config import DATA_PROCESSED, MODEL_DIR

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


def evaluate():

    print("Loading dataset...")

    X = joblib.load(os.path.join(DATA_PROCESSED, "X.joblib"))
    y = joblib.load(os.path.join(DATA_PROCESSED, "y.joblib"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.33,
        random_state=42
    )

    X_test = np.expand_dims(X_test, axis=2)

    print("Loading trained model...")

    model = load_model(os.path.join(MODEL_DIR, "emotion_model.h5"))

    print("Running predictions...")

    predictions = model.predict(X_test)
    predictions = np.argmax(predictions, axis=1)

    # Classification report
    report = classification_report(y_test, predictions, target_names=EMOTIONS)

    print("\nClassification Report\n")
    print(report)

    os.makedirs("reports", exist_ok=True)

    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTIONS,
        yticklabels=EMOTIONS
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig("reports/confusion_matrix.png")

    print("\nConfusion matrix saved to reports/")

def plot_training_curves():

    history_path = os.path.join("reports", "training_history.json")

    if not os.path.exists(history_path):
        print("Training history not found.")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    # Accuracy plot
    plt.figure()

    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="validation")

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig("reports/model_accuracy.png")

    # Loss plot
    plt.figure()

    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="validation")

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.savefig("reports/model_loss.png")

    print("Training curves saved.")

if __name__ == "__main__":
    evaluate()
    plot_training_curves()