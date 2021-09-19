from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from pose_estimation.human_action_recognition.helper import actions


def make_model():
    model = Sequential()
    # input shape is based on total number of landmarks extracted for 30 frames. Please see extract landmarks function in helper for more details
    model.add(
        LSTM(64, return_sequences=True, activation="relu", input_shape=(30, 1662))
    )
    model.add(LSTM(128, return_sequences=True, activation="relu"))
    model.add(LSTM(64, return_sequences=False, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(actions.shape[0], activation="softmax"))
    return model
