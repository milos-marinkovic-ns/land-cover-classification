from abc import ABC, abstractmethod
import numpy as np

class ClassificationModel(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def load(self, path: str):
        """
        Loads pretrained model from specific path
        path: str - Path you specify

        Returns:
        None
        """
        pass

    @abstractmethod
    def train(self, X_train: np.array, y_train: np.array) -> None:
        """
        Trains model on with given data

        Paramters:
        X_train: np.array - Features
        y_train: np.array - labels
        """
        pass

    @abstractmethod
    def predict(self, X_test: np.array) -> np.array:
        """
        Tests model on given dataset

        Parameters:
        X_test: np.array - Features
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Saves model to specific path

        Parametrs:
        path: str - Path where model will be saved
        """
        pass
