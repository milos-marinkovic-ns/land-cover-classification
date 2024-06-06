import pandas as pd
import numpy as np
import rasterio
import datetime
import os
from multiprocessing.dummy import Pool as ThreadPool
from src.utils.vegetation_index_util import VegetationIndicesCalculator
from src.modules.feature_extraction_module import FeatureExtraction
from src.models.classification_models.ClassificationModel import ClassificationModel
from src.models.classification_models.ClassficationModelFactory import ClassificationModelFactory
from typing import Generator
import json
from tqdm import tqdm

class MaskMaker:

    __sentinel_channels_config: str = 'crop-classification\configs\sentinel_configurations\sentinel_channels.json'
    __feature_extractors: list[Generator[pd.DataFrame, None, None]] = None


    def __init__(self, path_to_sentinel: str, dates: list[str], routes: list[str], output_mask_path: str, model_path: str, projection: str, color_map: np.array) -> None:
        self.path_to_sentinel: str = path_to_sentinel
        self.dates: list[str] = dates
        self.routes: routes[str] = routes
        self.output_mask_path: str = output_mask_path
        self.granules: list[str] = self.load_granule_names()
        self.model_metadata: dict = self.load_model_metadata(model_path)
        self.model: ClassificationModel = ClassificationModelFactory(model_name= self.model_metadata['model'], params= self.model_metadata['model_params']).create_model()
        self.projection: str = projection
        self.model.load(f"{model_path}//model.joblib")
        self.create_feature_extractors()
        self.metadata_granules_info: list[tuple[tuple[int, int], rasterio.Affine, str]] = self.get_granules_metadata()
        self.__custom_colormap = np.array(color_map)
        formatted_time = datetime.datetime.now().strftime(r"%Y_%m_%d_%H_%M")
        self.save_path = f"{self.output_mask_path}/masks_{formatted_time}"
        os.mkdir(self.save_path)


    def color_image(self, image: np.array) -> np.array:
        """
        Apllys color map to grayscale image

        Args:
        image: np.array - image to color

        Returns
        np.array - Colored image
        """
        image: np.array = image.astype(np.uint8)
        return self.__custom_colormap[image]

    def load_granule_names(self) -> list[str]:
        """
        Loads granule names

        Args:
        None

        Returns:
        names - list[str]
        """
        return json.load(open(self.__sentinel_channels_config))['routes'][self.routes[0]]
    
    def load_model_metadata(self, model_path: str) -> dict:
        """
        loads metadata

        Args:
        model_path: str - path to model directory

        Returns:
        dict - model_metadata
        """
        return json.load(open(f"{model_path}//metadata.json"))
    
    def create_feature_extractors(self) -> None:
        """
        Creates list of FeatureExtraction.get_all_features() generators

        Args:
        None

        Returns:
        None
        """
        self.__feature_extractors = [FeatureExtraction(
        gdf_path=None, 
        date=date, 
        route=route, 
        output_path=None,
        path_to_sentinel=self.path_to_sentinel,
        projection=self.projection)
        .get_all_features(True)
        for date, route in zip(self.dates, self.routes)]
        
    def get_granules_metadata(self) -> None:
        """
        Creates list of FeatureExtraction.granules_metadata() generators

        Args:
        None

        Returns:
        None
        """
        granules_metadata = FeatureExtraction(
        gdf_path=None, 
        date=self.dates[0], 
        route=self.routes[0], 
        output_path=None,
        path_to_sentinel=self.path_to_sentinel,
        projection=self.projection).granules_metadata()
        print("Reading metadata....")
        return [el for el in granules_metadata]


    def get_data_chunks(self, granule_dates: list[pd.DataFrame], chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """
        Generator for creating data chunk ready for prediction
        
        Args:
        granule_dates: list[pd.DataFrame]
        chunk_size: int

        Return:
        Generator[pd.DataFrame, None, None]
        """
        num_granules = len(granule_dates)
        num_rows = len(granule_dates[0])
    
        for start_idx in range(0, num_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, num_rows)
            data_to_merge = self.get_vegetation_idices_for_data_chunks(start_idx, end_idx, num_granules, granule_dates)
            merged_data = pd.concat(data_to_merge, axis=1)
            yield merged_data[:chunk_size]

    def get_vegetation_idices_for_data_chunks(self, start_idx: int, end_idx: int, num_granules: int, granule_dates: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Calculates vegetation indices from each date for each chunk of data

        Args:
        start_idx: int 
        end_idx: int
        num_granules: int
        granule_dates: list[pd.DataFrame]

        Returns:
        list[pd.DataFrame]

        """
        data_to_merge = []
        for date_idx in range(num_granules):
            data_chunk = granule_dates[date_idx].iloc[start_idx:end_idx].copy()
            vic: VegetationIndicesCalculator = VegetationIndicesCalculator(data_chunk, self.dates[date_idx], self.routes[date_idx])
            vic()
            data_to_merge.append(data_chunk)
        return data_to_merge

    def get_granule_from_each_date(self) -> list[pd.DataFrame]:
        """
        Combines dataframes of granules from each data

        Args:
        None

        Returns:
        list[pd.DataFrame] - Granules dataframes from each data
        """
        granules_from_each_data = []
        for feature_extractor in self.__feature_extractors:
            df = next(feature_extractor)
            granules_from_each_data.append(df)
        return granules_from_each_data
    
    def validate_predictions(self, predictions: np.array, prediction_len: int) -> bool:
        """
        Validates if prediction is of excpected size

        Args:
        predictions: np.array
        prediction_len: int

        Returns:
        bool
        """
        return predictions.shape == (prediction_len, )

    
    def make_prediction(self, data_chunk: pd.DataFrame, prediction_len: int) -> np.array:
        """
        Makes prediction from loaded data_chunk

        Args:
        data_chunk: pd.DataFrame
        prediction_len: int

        Returns:
        np.array
        """
        data_chunk.fillna(0, inplace=True)
        feature_matrix = data_chunk.values
        feature_matrix = np.clip(feature_matrix, -1e6, 1e6)
        predictions = self.model.predict(feature_matrix)
        return predictions if self.validate_predictions(predictions, prediction_len) else np.zeros((prediction_len,))

    def get_single_mask(self, mask_shape: tuple[int, int], granule_from_each_data: list[pd.DataFrame]) -> np.array:
        """
        Creates single mask as making predictions row by row from sentinel image 

        Args:
        mask_shape: tuple[int, int] - Shape of sentinel image
        granule_from_each_data: list[pd.Dataframe] 

        Returns:
        list[pd.Data]
        """
        final_mask = np.zeros(mask_shape)
        with tqdm(total=mask_shape[0]) as pbar:
            for row_idx, data_chunk in tqdm(enumerate(self.get_data_chunks(granule_from_each_data, mask_shape[0]))):
                predictions = self.make_prediction(data_chunk, mask_shape[0])
                final_mask[row_idx] = predictions
                pbar.update(1)

        return final_mask
    
    def save_image(self, output_file_path: str, image: np.array, affine_transform: rasterio.Affine) -> None:
        """
        Saves transformed image 

        Args:
        output_file_path: str
        image: np.array
        affine_transform: rasterio.Affine

        Returns: 
        None
        """
        with rasterio.open(
            output_file_path,
            'w',
            driver='GTiff',
            height=image.shape[0],
            width=image.shape[1],
            count=image.shape[2], 
            dtype=image.dtype,
            crs=self.projection,
            transform=affine_transform,
        ) as dst:
            for band in range(image.shape[2]):
                dst.write(image[:, :, band], band + 1)

    def get_all_masks(self):
        """
        Iterates over all granules inside sentinel directory and gets its feature, then creats model crops predictions

        Args:
        None
        Returns:
        None
        """
        for granule_info in self.metadata_granules_info:
            granule_name: str = granule_info[-1]
            print(f"Creating mask for granule: {granule_name}")
            mask_shape: tuple[int, int] = granule_info[0]
            granule_from_each_date: list[pd.DataFrame] = self.get_granule_from_each_date()
            mask_prediction: np.array =  self.get_single_mask(mask_shape, granule_from_each_date)
            save_mask_path: str = self.save_path + f"/{granule_name}.tif" 
            image_to_save: np.array = self.color_image(mask_prediction)
            self.save_image(save_mask_path, image_to_save, granule_info[-2])

        