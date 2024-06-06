import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from typing import Generator
from shapely.geometry import Point
from rasterio.mask import geometry_mask
from src.utils.raster_image_util import resize_image
from src.utils.raster_image_util import adjust_transform
from src.utils.raster_image_util import open_raster_image
from src.utils.sentinel_image_paths_util import get_images_paths
from src.utils.propmt_util import prompt_current_image_processing


class FeatureExtraction:

    def __init__(self, gdf_path: str, date: str, route: str, output_path: str, path_to_sentinel: str, projection: str) -> None:
        self.__gdf: str = gpd.read_file(gdf_path) if gdf_path is not None else None
        self.__date: str = date
        self.__route: str = route
        self.__output_path: str = output_path
        self.__data: dict = {}
        self.__target_shape: tuple = None
        self.__target_transform: rasterio.Affine = None
        self.__band: str = None
        self.__resolution: str = None
        self.__granule: str = None
        self.__image: np.array = None
        self.__transform: np.array = None
        self.__replacement_value: int = None
        self.__path_to_sentinel: str = path_to_sentinel
        self.__projection: str = projection
    
    def reset_targets(self):
        """
        Resets target values

        Parameters:
        None

        Returns:
        None
        """
        self.__data = {}
        self.__target_shape = None
        self.__target_transform = None

    def adjust_image_resolution(self) -> None:
        """
        Checking resolution of image and resizing it if its neccessery

        Parameters:
        None
        Returns:
        None
        """
        if self.__resolution == "10m" and not self.__target_shape and not self.__target_transform:
            self.__target_shape = self.__image.shape
            self.__target_transform = self.__transform
        elif self.__resolution != "10m":
            self.__image = resize_image(self.__image, self.__target_shape)
            self.__transform = adjust_transform(self.__transform, self.__target_transform)


    def extract_labeled_pixel_values(self, paths: list[str]) -> None:
        """
        Extracts values from labeled pixels of each channel

        Parameters:
        paths: list[str] - Paths of each channel/band

        Returns:
        None
        """

        for path in paths:
            self.__band, self.__resolution, self.__granule = prompt_current_image_processing(path, self.__date, self.__route)
            self.__image, self.__transform = open_raster_image(path)
            self.adjust_image_resolution()
            mask_value: np.array = geometry_mask(self.__gdf.geometry, transform=self.__transform, invert=True, out_shape=self.__image.shape)
            self.__replacement_value = np.max(self.__image) + 1
            self.__image[~mask_value] = self.__replacement_value
            pixel_values = self.__image[self.__image != self.__replacement_value]
            self.__data[f"{self.__band}_{self.__date}_{self.__route}"] = pixel_values

    def get_image_metadata(self, path: str) -> tuple[tuple[int, int], rasterio.Affine]:
        """
        Extract image transformation and size from path

        Args:
        path: str - Path to image

        Returns:
        Image shape and image transformation
        """
        self.__band, self.__resolution, self.__granule = prompt_current_image_processing(path, self.__date, self.__route)
        self.__image, self.__transform  = open_raster_image(path)
        granule_name = path.split("\\")[-1].split("_")[0]
        return self.__image.shape, self.__transform, granule_name



    def extract_all_pixel_values(self, paths: list[str], making_masks: bool = False) -> pd.DataFrame:
        """
        Extracts all features for images ready to be used for creating map

        Parameters:
        paths: list[str] - Paths to each channel/band

        Returns:
        pd.DataFrame
        """
        for path in paths:
            self.__band, self.__resolution, self.__granule = prompt_current_image_processing(path, self.__date, self.__route)
            self.__image, self.__transform  = open_raster_image(path)
            self.adjust_image_resolution()
            self.__data[f"{self.__band}_{self.__date}_{self.__route}"] = self.__image.flatten()
            result = pd.DataFrame(self.__data)
        if making_masks:
            return result
        else: 
            return result[
                (result[f"B02_{self.__date}_{self.__route}"] != 0) &
                (result[f"B03_{self.__date}_{self.__route}"] != 0) &
                (result[f"B04_{self.__date}_{self.__route}"] != 0)
            ]

    def get_world_coordinates(self):
        """
        Translating image coordinates to world coordinates

        Parameters:
        None
        Returns:
        None
        """
        print("Creating points...")
        row_indices, col_indices = np.where(self.__image != self.__replacement_value)
        points = [Point(self.__transform*(col, row)) for row, col in zip(row_indices, col_indices)]
        print("Points created")
        self.__data['geometry'] = points

    def save_processed_granule(self):
        new_gdf = gpd.GeoDataFrame(self.__data)
        new_gdf.crs = self.__projection
        result = gpd.sjoin(new_gdf, self.__gdf, how='inner', predicate='intersects').drop(columns = "index_right")
        result = result[
                (result[f"B02_{self.__date}_{self.__route}"] != 0) &
                (result[f"B03_{self.__date}_{self.__route}"] != 0) &
                (result[f"B04_{self.__date}_{self.__route}"] != 0)
            ]
        result.to_csv(f"{self.__output_path}/features_{self.__date}_{self.__route}/{self.__granule}_{self.__date}_{self.__route}_features.csv")



    def get_labeled_features(self) -> None:
        """
        Extracting features from .jp2 images on a single data and from a single route inside data/sentinel folder labeld by ground_truth file
        Parameters:
        None
        Returns:
        None
        """
        os.mkdir(f"{self.__output_path}/features_{self.__date}_{self.__route}")
        for paths in tqdm(get_images_paths(self.__date, self.__route, self.__path_to_sentinel)):
            self.reset_targets()
            self.extract_labeled_pixel_values(paths)
            self.get_world_coordinates()
            self.save_processed_granule()

    def get_all_features(self, making_masks: bool = False) -> Generator[pd.DataFrame, None, None]:
        """
        Extracting features from .jp2 images on a single data and from a single route inside data/sentinel folder
        Parameters:
        None
        Returns:
        None
        """
        for paths in tqdm(get_images_paths(self.__date, self.__route, self.__path_to_sentinel)):
            self.reset_targets()
            yield self.extract_all_pixel_values(paths, making_masks)

    def granules_metadata(self):
        """
        Extracting metadata from .jp2 images on a single data and from a single route inside data/sentinel folder
        Parameters:
        None
        Returns:
        None
        """
        for paths in tqdm(get_images_paths(self.__date, self.__route, self.__path_to_sentinel)):
            yield self.get_image_metadata(paths[0])
            