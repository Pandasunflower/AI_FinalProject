import os
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from ..base import BaseModel
from utils import curve

class DNN(BaseModel, ABC):
    """
    Base class for all Keras-based deep learning models

    Args:
        n_classes (int): the number of labal(emotion)
        lr (float): learning rate
    """
    def __init__(self, model: Sequential, trained: bool = False) -> None:
        super(DNN, self).__init__(model, trained)
        print(self.model.summary())

    def save(self, path: str, name: str) -> None:
        """
        save the model

        Args:
            path (str): the path to save the model
            name (str): the file name of the model
        """
        h5_save_path = os.path.join(path, name + ".h5")
        self.model.save_weights(h5_save_path)

        save_json_path = os.path.join(path, name + ".json")
        with open(save_json_path, "w") as json_file:
            json_file.write(self.model.to_json())

    @classmethod
    def load(cls, path: str, name: str):
        """
        load the model

        Args:
            path (str): the path to load the model
            name (str): the file name of the model
        """
        # load json
        model_json_path = os.path.abspath(os.path.join(path, name + ".json"))
        json_file = open(model_json_path, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        # load weight
        model_path = os.path.abspath(os.path.join(path, name + ".h5"))
        model.load_weights(model_path)

        return cls(model, True)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: int = 32,
        n_epochs: int = 20
    ) -> None:
        """
        train the model

        Args:
            x_train (np.ndarray): train dataset sample
            y_train (np.ndarray): train dataset label
            x_val (np.ndarray, optional): test dataset sample
            y_val (np.ndarray, optional): test dataset label
            batch_size (int): Batch size(seperate a epoch in to n batch)
            n_epochs (int): Epoch(seperate a data in to n epoch)
        """
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        x_train, x_val = self.reshape_input(x_train), self.reshape_input(x_val)

        history = self.model.fit(
            x_train, y_train,
            batch_size = batch_size,
            epochs = n_epochs,
            shuffle = True,  # random rearrange the training data before every epoch
            validation_data = (x_val, y_val)
        )

        # the loss and the accuracy of train dataset
        acc = history.history["accuracy"]
        loss = history.history["loss"]
        # the loss and the accuracy of test dataset
        val_acc = history.history["val_accuracy"]
        val_loss = history.history["val_loss"]

        curve(acc, val_acc, "Accuracy", "acc")
        curve(loss, val_loss, "Loss", "loss")

        self.trained = True

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        predict the emotion of the audio feature

        Args:
            samples (np.ndarray): audio feature

        Returns:
            results (np.ndarray): the result of the prediction
        """
        # if the model hasn't been train yet
        if not self.trained:
            raise RuntimeError("There is no trained model.")

        samples = self.reshape_input(samples)
        return np.argmax(self.model.predict(samples), axis=1)

    def predict_proba(self, samples: np.ndarray) -> np.ndarray:
        """
        predict the probability of every emotion

        Args:
            samples (np.ndarray): audio feature

        Returns:
            results (np.ndarray): the probability of every emotion
        """
        if not self.trained:
            raise RuntimeError('There is no trained model.')

        if hasattr(self, 'reshape_input'):
            samples = self.reshape_input(samples)
        return self.model.predict(samples)[0]

    @abstractmethod
    def reshape_input(self):
        pass
