from src.modules.trainer_module import ClassificationTrainer

def train_model(model_name: str, model_parameters: dict, train_features_path: str, output_path: str, model_directory_path: str, columns_to_exclude: str, target_column: str):
    classfication_trainer: ClassificationTrainer = ClassificationTrainer(model_params= model_parameters, model_name= model_name, train_features_path=train_features_path, model_path=model_directory_path, target_column=target_column, columns_to_exclude=columns_to_exclude)
    classfication_trainer.train(save_model=True, output_path=output_path)


def test_model(test_features_path: str, model_directory_path: str, model_report_path: str, columns_to_exclude: str, target_column: str):
    classfication_trainer: ClassificationTrainer = ClassificationTrainer(model_path=model_directory_path, test_features_path=test_features_path, model_report_path=model_report_path, target_column=target_column, columns_to_exclude=columns_to_exclude)
    classfication_trainer.test()