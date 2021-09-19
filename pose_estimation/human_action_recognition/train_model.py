from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from pose_estimation.human_action_recognition.model import make_model
from pose_estimation.human_action_recognition.helper import (
    actions,
    num_sequences,
    sequence_length,
    DATA_PATH,
)
import numpy as np
import os
import tensorflow as tf


def train_model():
    model = make_model()
    # extract data and split into train and test
    label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(num_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(
                    os.path.join(
                        DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)
                    )
                )
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
        X = np.array(sequences)
        Y = to_categorical(labels).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05)

    try:
        os.makedirs("logs/weights/")
    except Exception:
        pass
    checkpoint_filepath = "logs/weights/"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="categorical_accuracy",
        mode="max",
        save_best_only=True,
    )
    model.compile(
        optimizer="Adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    model.fit(X_train, y_train, epochs=2000, callbacks=[model_checkpoint_callback])
