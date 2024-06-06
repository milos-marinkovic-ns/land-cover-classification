from src.modules.feature_extraction_module import FeatureExtraction


def feature_extraction_for_single_date(gdf_path: str, date: str, route: str, output_path: str, path_to_sentinel, projection: str) -> None:
    feature_extraction: FeatureExtraction = FeatureExtraction(gdf_path, date, route, output_path, path_to_sentinel, projection)
    feature_extraction.get_labeled_features()

def feature_extraction_for_multiple_dates(gdf_path: str, dates: list[str], routes: list[str], output_path: str, path_to_sentinel: str, projection:str) -> None:
    for date, route in zip(dates, routes):
        print(date, route)
        feature_extraction: FeatureExtraction = FeatureExtraction(gdf_path, date, route, output_path, path_to_sentinel, projection)
        feature_extraction.get_labeled_features()
