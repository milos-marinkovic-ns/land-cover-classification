import datetime
import os
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.vegetation_index_util import VegetationIndicesCalculator
from sklearn.metrics import classification_report, confusion_matrix
from src.models.classification_models.ClassficationModelFactory import ClassificationModelFactory
from src.models.classification_models.ClassificationModel import ClassificationModel

class ClassificationTrainer:

    def __init__(self, test_features_path: str = None, train_features_path: str = None, model_path: str = None, model_name: str = None, model_params: dict = None, model_report_path: str = None, columns_to_exclude: list[str] = None, target_column: str = None) -> None:
        """
        ClassificationTrainer trains and tests machine learning models on data extracted by Feature extraction module

        Parametrs:
        model_path: str - Optional if you want to load pretrained model to do some fine tuning or testing
        model_name: str - Available model names: "RF" - Random Forrest, "SVM" - Support Vector Machine, "XGB" - XGBoost
        model_params: dict - Hyperparametrs of model
        train_features_path: str - Path to train features extracted by feature extraction module 
        test_features_path: str - Patho to test features extracted by feature extraction module 
        model_report_path: str - Path to export report and confusion matrix
        """
        self.__model_name = model_name
        if not model_path == None:
            model_metadata: dict = json.load(open(f"{model_path}//metadata.json"))
            self.model: ClassificationModel  = ClassificationModelFactory(model_name= model_metadata['model'], params=model_metadata['model_params']).create_model()
            self.model.load(f"{model_path}//model.joblib")
        elif not (model_name == None or model_params == None):
            self.model: ClassificationModel = ClassificationModelFactory(model_params, model_name).create_model()
        self.__train_features_path: str = train_features_path
        self.__test_features_path: str = test_features_path
        self.__model_report_path: str = model_report_path
        self.__metadata: dict = {
            "dates": [],
            "granule": [],
            "model": model_name,
            "model_params": model_params
        }
        self.columns_to_exclude: list[str] = columns_to_exclude
        self.target_column: str = target_column

    def __get_metadata_dates(self) -> str:
        """
        Getting all dates used for training
        """
        for file_name in os.listdir(self.__train_features_path):
            tokens: list = file_name.split("_")
            date: str = tokens[1]
            self.__metadata['dates'].append(date)
        return file_name

    def __get_metadata_granules(self, file_name: str) -> None:
        """
        Getting all granules used for training
        """
        for granule in os.listdir(self.__train_features_path + f"/{file_name}"):
            tokens:str = granule.split("_")
            granule: str = tokens[0]
            self.__metadata['granule'].append(granule)


    def __get_trainer_metadata(self):
        """
        Creating metadata of trainer
        """
        granule_filename:str =  self.__get_metadata_dates()
        self.__get_metadata_granules(granule_filename)
        print("Training model on this data:\n", self.__metadata)


    def __check_if_all_dates_have_same_number_of_granule(self, features_path: str) -> bool:
        """
        Checks if there is a same number of granules in each data before it starts training

        Parameters:
        features_path: str - Path to folder of all features

        Returns:
        bool
        """
        last_number_of_granules: int = None
        same_number: bool = True
        for name in os.listdir(features_path):
            folder_path = features_path + "\\" + name
            current_number_of_granules = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
            if last_number_of_granules == None:
                last_number_of_granules = current_number_of_granules
            else:
                same_number = current_number_of_granules == last_number_of_granules
        return same_number
    

    def __load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads features from feature file

        Parameters:
        file_path: str - Feature 

        Returns:
        pd.DataFrame
        """
        date:str = file_path.split("_")[-3]
        route:str = file_path.split("_")[-2]
        dataframe: pd.DataFrame = pd.read_csv(file_path)
        vegetation_indices_calculator: VegetationIndicesCalculator = VegetationIndicesCalculator(dataframe,date,route)
        vegetation_indices_calculator()
        return dataframe


    def __preprocess_dataframe(self, dataframe: pd.DataFrame) -> tuple[np.array, np.array]:
        """
        Removes unwanted data from dataframe and returns data ready for training

        Parameters:
        dataframe: pd.Dataframe - Dataframe to process
        
        Returns:
        tuple[np.array, np.array] -> X_train, y_train
        """
        columns_to_exclude = self.columns_to_exclude
        dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)
        dataframe.dropna(inplace=True)
        x_train = dataframe.drop(columns=columns_to_exclude).to_numpy()
        y_train = dataframe[self.target_column].to_numpy()
        return x_train, y_train
    

    def __save_model(self, output_path: str) -> None:
        """
        Saves model and its metadata

        output_paht: str - Ouput directory
        """
        formatted_time = datetime.datetime.now().strftime(r"%Y_%m_%d_%H_%M")
        save_path = f"{output_path}/{self.__model_name}_{formatted_time}"
        os.mkdir(save_path)
        print(save_path)
        self.model.save(f"{save_path}/model.joblib")
        with open(f"{save_path}/metadata.json", "w") as fp:
            json.dump(self.__metadata, fp)

        fp.close()

    def __get_all_data_from_same_granule(self, directories: list[str], nth_granule: int, features_path: str) -> pd.DataFrame:
        """
        Combines all dates that are from same granule

        Parameters:
        directories: list[str] - Directories of diffrent dates
        nth_granule: int - Ordinal number of granule in each directory
        features_path: str - Current features
        
        Returns:
        None
        """
        df_list: list = []
        for directory in directories:
            file_path = os.path.join(features_path, directory, os.listdir(os.path.join(features_path, directory))[nth_granule])
            dataframe = self.__load_data(file_path)
            df_list.append(dataframe)

        granule:str = file_path.split("_")[-4]
        print(f"Current on: {granule}")
        merged_df = pd.concat(df_list, axis=1)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
        return merged_df

    def train(self, save_model: bool = True, output_path: str = "."):
        """
        Trains model defined in constructor

        Parameters:
        save_model: bool (Default = True)
        output_path: str (Default = "model.joblib")

        Returns:
        None
        """
        X_train_list = []
        y_train_list = []

        if not self.__check_if_all_dates_have_same_number_of_granule(self.__train_features_path):
            print("In all directories should be same numbe of granules")
            return
        directories = [d for d in os.listdir(self.__train_features_path) if os.path.isdir(os.path.join(self.__train_features_path, d))]
        self.__get_trainer_metadata()
        for nth_granule in range(len(os.listdir(os.path.join(self.__train_features_path, directories[0])))):
            merged_df = self.__get_all_data_from_same_granule(directories, nth_granule, self.__train_features_path)
            merged_df = merged_df.drop(columns=['fid']) if 'fid' in merged_df.columns else merged_df
            X_train, y_train = self.__preprocess_dataframe(merged_df)
            X_train_list.append(X_train)
            y_train_list.append(y_train)

            overall_X_train: np.array = np.vstack(X_train_list)
            overall_y_train: np.array = np.concatenate(y_train_list)
    
        self.model.train(overall_X_train, overall_y_train)
        if save_model:
            self.__save_model(output_path)

    def __save_classification_report(self, y_test: np.array, y_pred: np.array) -> None:
        """
        Saving classification report

        Parameters

        y_test: np.array
        y_pred: np.array
        """
        report = classification_report(y_test, y_pred)
        with open(f'{self.__model_report_path}//classification_report.txt', 'w') as report_file:
            report_file.write(report)


    def __save_confusion_matrix(self, y_test: np.array, y_pred: np.array) -> None:
        """
        Saving confusion matrix

        Parameters

        y_test: np.array
        y_pred: np.array
        """
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_pred))
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(f'{self.__model_report_path}//confusion_matrix.png')


    def test(self):
        """
        Testing classficiation model
        """
        print("Waiting for test results, be patinent!")
        if not self.__check_if_all_dates_have_same_number_of_granule(self.__test_features_path):
            print("In all directories should be same numbe of granules")
            return
        directories = [d for d in os.listdir(self.__test_features_path) if os.path.isdir(os.path.join(self.__test_features_path, d))]
        y_test_all = np.array([])
        y_pred_all = np.array([])
        for nth_granule in range(len(os.listdir(os.path.join(self.__test_features_path, directories[0])))):
            merged_df = self.__get_all_data_from_same_granule(directories, nth_granule, self.__test_features_path)
            merged_df = merged_df.drop(columns=['fid']) if 'fid' in merged_df.columns else merged_df
            X_test, y_test = self.__preprocess_dataframe(merged_df)
            if X_test.shape[0] == 0:
                continue
            y_pred = self.model.predict(X_test)
            y_test_all = np.concatenate((y_test_all, y_test.astype(np.uint8)))
            y_pred_all = np.concatenate((y_pred_all, y_pred.astype(np.uint8)))

        self.__save_classification_report(y_test_all, y_pred_all)
        self.__save_confusion_matrix(y_test_all, y_pred_all)