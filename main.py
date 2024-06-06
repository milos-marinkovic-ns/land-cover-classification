from src.exceptions.exceptions import UndefinedFunctionError
from src.services.smoketest import check_poetry_setup
from src.services.feature_extraction_service import (
    feature_extraction_for_single_date,
    feature_extraction_for_multiple_dates
)
from src.services.model_train_service import (
    train_model,
    test_model
)
from src.services.make_mask_service import (
    make_masks
)
import sys
import os
import json


def main():
    """
    Reading config file passed when running script, calling specific function from configuration folder
    """

    data_path: str = sys.argv[1]
    os.chdir("..")
    with open(data_path, 'r') as file:
        data:dict = json.load(file)

    function_name:str = data.get("function_to_call")
    parameters:dict = data.get("parameters", {})

    if function_name in globals():
        function_to_call: function = globals()[function_name]
        function_to_call(**parameters)
    else:
        raise UndefinedFunctionError(function_name)

if __name__ == "__main__":
    main()
