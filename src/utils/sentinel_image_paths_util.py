import json
import os
from typing import Generator

def get_sentinel_config() -> dict:
    """
    Returns sentinel config
    """
    with open("crop-classification\configs\sentinel_configurations\sentinel_channels.json", "r") as file:
        sentinel_config: dict = json.load(file)
        file.close()
    return sentinel_config

def get_granules_of_interest(date: str, route: str, path_to_data: str) -> list:
    """
    Gets names of directories of interest
    Paramaters:
    date: str - Same as in directory name
    route: str - Same as in directory name
    Returns:
    list: granules of interest
    """
    sentinel_config: dict = get_sentinel_config()
    dirs: list = [path for path in os.listdir(f"{path_to_data}\sentinel") 
     if date in path
     and route in path
     and any(substring in path for substring in sentinel_config['routes'][route])]
    
    return dirs

def get_all_bands_from_granule(granule_path: str) -> list:
    """
    Takes of bands of interest from single granule

    Parameters:
    granule_path: str - path to granule

    Returns:
    list: all paths to images from single granule
    """

    bands: list = []
    sentinel_config: dict = get_sentinel_config()
    for resolution in sentinel_config['resolutions']:
        resolution_dir:str = granule_path + f"\{resolution}"
        for band in sentinel_config['resolutions'][resolution]:
            image_names:str = os.listdir(resolution_dir)
            result:str = next(filter(lambda x: band in x, image_names), None)
            bands.append(resolution_dir + f"\{result}")

    return bands

def get_images_paths(date: str, route: str, path_to_sentinel: str) -> Generator:
    """
    Generator that returns granule by granule images of same date and route

    date: str - Date of shot
    route: str - Route of shot
    """
    for granule in get_granules_of_interest(date, route, path_to_sentinel):
        images_folder_name: str = os.listdir(f"{path_to_sentinel}\sentinel\{granule}\GRANULE")[0]
        bands: list = get_all_bands_from_granule(f"{path_to_sentinel}\sentinel\{granule}\GRANULE\{images_folder_name}\IMG_DATA")
        yield bands
    