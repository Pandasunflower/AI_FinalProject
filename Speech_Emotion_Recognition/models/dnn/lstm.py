from tensorflow.keras.layers import LSTM as KERAS_LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
from .dnn import DNN

class LSTM(DNN):
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(LSTM, self).__init__(model, trained)

    @classmethod
    def make(
        cls,
        input_shape: int,
        rnn_size: int,
        hidden_size: int,
        dropout: float = 0.5,
        n_classes: int = 3,
        lr: float = 0.001
    ):
        """
        build the model

        Args:
            input_shape (int): the dimension of the feature
            rnn_size (int): the size of the LSTM hidden layer
            hidden_size (int): the size of the full connected layer
            dropout (float, optional, default=0.5): dropout
            n_classes (int, optional, default=6): the number of labal(emotion)
            lr (float, optional, default=0.001): learning rate
        """
        model = Sequential()

        model.add(KERAS_LSTM(rnn_size, input_shape=(1, input_shape)))  # (time_steps = 1, n_feats)
        model.add(Dropout(dropout))
        model.add(Dense(hidden_size, activation='relu'))
        # model.add(Dense(rnn_size, activation='tanh'))

        model.add(Dense(n_classes, activation='softmax'))  # classification layer
        optimzer = Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimzer, metrics=['accuracy'])

        return cls(model)

    def reshape_input(self, data: np.ndarray) -> np.ndarray:
        """reshape the input from 2D to 3D"""
        # (n_samples, n_feats) -> (n_samples, time_steps = 1, input_size = n_feats)
        # time_steps * input_size = n_feats
        data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        return data
