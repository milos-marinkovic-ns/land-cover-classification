from src.models.classification_models.ClassificationModel import ClassificationModel
from .RandomForrestModel import RandomForestModel
from .SupportVectorMachineModel import SupportVectorMachineModel
from .XGBoostModel import XGBoostModel


class ClassificationModelFactory:
    def __init__(self, params: dict, model_name: str) -> ClassificationModel:
        """
        Creates ClassificationModel object based on model and name and its params

        params: dict - model paramters
        model_name: str - model name
        """
        model_class = globals().get(model_name + "Model")
        if model_class:
            self.model: ClassificationModel = model_class(params)
        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def create_model(self):
        return self.model