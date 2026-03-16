from keras.models import Sequential
from keras.layers import Conv1D, Activation, Dropout, Flatten, Dense
def build_model():
    model = Sequential()
    model.add(
        Conv1D(
            128,
            5,
            padding="same",
            input_shape=(40,1)
        )
    )

    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    model.add(Conv1D(128,5,padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(8))
    model.add(Activation("softmax"))

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="rmsprop",
        metrics=["accuracy"]
    )
    return model