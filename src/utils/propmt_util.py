def prompt_current_image_processing(path: str, date: str, route: str) -> tuple[str]:
    """
    Prompts info of current image processing

    Parameters:
    path: str - Path of current image
    date: str - Date of current image
    route: str - Route of current image

    Returns:
    tuple - propmpt information
    """
    tokens = path.split("_")
    band: str = tokens[-2]
    resolution: str = tokens[-1].split(".")[0]
    granule: str = tokens[-4].split("\\")[-1]
    print(f"Processing band:{band}, resolution:{resolution}, date: {date}, route:{route}, granule:{granule}")
    return band, resolution, granule
