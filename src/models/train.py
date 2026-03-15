import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from src.models.cnn_model import build_model
from src.config.config import DATA_PROCESSED, MODEL_DIR
from sklearn.utils.class_weight import compute_class_weight

def train():

    X = joblib.load(DATA_PROCESSED + "/X.joblib")
    y = joblib.load(DATA_PROCESSED + "/y.joblib")

    classes = np.unique(y)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y
    )
    class_weights = dict(zip(classes, weights))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.33,
        random_state=42
    )

    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    model = build_model()

    history = model.fit(
        X_train,
        y_train,
        batch_size=16,
        epochs=120,
        validation_data=(X_test, y_test),
        class_weight=class_weights
    )

    predictions = model.predict(X_test)
    predictions = predictions.argmax(axis=1)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    model.save(MODEL_DIR + "/emotion_model.h5")


if __name__ == "__main__":
    train()