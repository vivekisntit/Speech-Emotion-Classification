from keras.models import Sequential
from keras.layers import Conv1D, Activation, Dropout, Flatten, Dense, MaxPooling1D


def build_model():

    model = Sequential()

    # Block 1
    model.add(
        Conv1D(
            128,
            5,
            padding="same",
            input_shape=(40, 1)
        )
    )
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv1D(128, 5, padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # Flatten
    model.add(Flatten())

    # Dense layer
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(8))
    model.add(Activation("softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model