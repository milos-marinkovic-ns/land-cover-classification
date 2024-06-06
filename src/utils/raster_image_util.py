from scipy.ndimage import zoom
import rasterio
import numpy as np
from rasterio.features import geometry_mask
import geopandas as gpd
from shapely import Point

def resize_image(src_data: np.array, target_shape: tuple) -> np.array:
    """
    Upsamples smaller images to a target size
    src_data: np.array - Image to resize
    target_shape: tuple - Shape size to transform to
    """

    return zoom(src_data, (
        target_shape[0] / src_data.shape[0],
        target_shape[1] / src_data.shape[1]
    ), order=1)


def adjust_transform(src_transform: np.array, target_transform: np.array) -> rasterio.Affine:
    """
    Adjusts transofmr matrix

    src_transform: np.array - Transform matrix to resize
    target_transform: np.array - Transform matrix to adjust to
    """
    return rasterio.Affine(
        target_transform.a,
        src_transform.b,
        target_transform.c,
        target_transform.d,
        target_transform.e,
        target_transform.f
    )

def open_raster_image(path: str) -> tuple:
    """
    Opens raster image from path

    Parameters:
    path: str - Path to raster image

    Returns:
    tuple (np.array, raster.Affine) - image and transformation
    """
    with rasterio.open(path) as src:
        image: np.array = src.read(1)
        transform: rasterio.Affine = src.transform
    return image, transform

