from src.models.classification_models.ClassificationModel import ClassificationModel
from sklearn.svm import SVC
import numpy as np
import joblib


class SupportVectorMachineModel(ClassificationModel):

    def __init__(self, params: dict = None) -> None:
        super().__init__()
        if params is None:
            self.__model = None
        else:
            self.__model = SVC(**params)

    def __str__(self) -> str:
        if self.__model is not None:
            return str(self.__model.get_params())
        else:
            return "Model not initialized yet. Load model or call constructor with parameters!"

    def load(self, path: str):
        try:
            self.__model = joblib.load(path)
        except FileNotFoundError:
            print(f"Error: Model file not found at {path}")

    def train(self, X_train: np.array, y_train: np.array):
        if self.__model is not None:
            self.__model.fit(X_train, y_train)
        else:
            print("Error: Model not initialized. Load model or call constructor with parameters!")

    def predict(self, X_test: np.array) -> np.array:
        if self.__model is not None:
            return self.__model.predict(X_test)
        else:
            print("Error: Model not initialized. Load model or call constructor with parameters!")

    def save(self, path: str):
        if self.__model is not None:
            joblib.dump(self.__model, path)
        else:
            print("Error: Model not initialized. Train the model before saving.")