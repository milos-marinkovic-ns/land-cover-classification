from src.modules.make_mask_module import MaskMaker


def make_masks(path_to_sentinel: str, dates: list[str], routes: list[str], output_path: str, model_path: str, projection: str, color_map: list[list[int]]):
    mask_maker: MaskMaker = MaskMaker(path_to_sentinel=path_to_sentinel, dates=dates, routes=routes, output_mask_path=output_path, model_path=model_path, projection=projection, color_map=color_map)
    mask_maker.get_all_masks()