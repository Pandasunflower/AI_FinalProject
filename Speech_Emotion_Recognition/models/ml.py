import os
import pickle
from abc import ABC
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator
import joblib
from .base import BaseModel

class MLModel(BaseModel, ABC):
    def __init__(self, model: BaseEstimator, trained: bool = False) -> None:
        super(MLModel, self).__init__(model, trained)

    def save(self, path: str, name: str) -> None:
        """
        save the model

        Args:
            path (str): the path to save the model
            name (str): the file name of the model
        """
        save_path = os.path.abspath(os.path.join(path, name + '.m'))
        pickle.dump(self.model, open(save_path, "wb"))

    @classmethod
    def load(cls, path: str, name: str):
        """
        load the model

        Args:
            path (str): the path to load the model
            name (str): the file name of the model
        """
        model_path = os.path.abspath(os.path.join(path, name + '.m'))
        model = joblib.load(model_path)
        return cls(model, True)

    def train(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        train the model

        Args:
            x_train (np.ndarray): train dataset sample
            y_train (np.ndarray): train dataset label
        """
        self.model.fit(x_train, y_train)
        self.trained = True

    def predict(self, samples: np.ndarray) -> np.ndarray:
        """
        predict the emotion of the audio feature

        Args:
            samples (np.ndarray): audio feature

        Returns:
            results (np.ndarray): the result of the prediction
        """
        if not self.trained:
            raise RuntimeError('There is no trained model.')
        return self.model.predict(samples)

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
        return self.model.predict_proba(samples)[0]


class SVM(MLModel):
    def __init__(self, model: BaseEstimator, trained: bool = False) -> None:
        super(SVM, self).__init__(model, trained)

    @classmethod
    def make(cls, params):
        model = SVC(**params)
        return cls(model)


class MLP(MLModel):
    def __init__(self, model: BaseEstimator, trained: bool = False) -> None:
        super(MLP, self).__init__(model, trained)

    @classmethod
    def make(cls, params):
        model = MLPClassifier(**params)
        return cls(model)
