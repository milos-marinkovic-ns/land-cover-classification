# Land cover classification

This repositry contains modules for crop classification from Sentinel 2 images.

## Installation and setting up the envoirment

First, you need to install Poetry. To accomplish this, execute the following command:

```bash
pip install poetry
```

Afterward, clone this repository by using the `git clone` command:

```bash
git clone http://git.biosens.rs/milos.marinkovic/crop-classification
```

Now, navigate to this repository by executing the following command:

```bash
cd crop-classification
```

Subsequently, execute the following command:

```bash
poetry config virtualenvs.in-project true
poetry install
```

This should generate a .venv file in your repository. Activate the virtual environment using:

```bash
.venv\Scripts\activate
```

To verify if everything is functioning correctly, execute the `testsmoke` command. This procedure will be employed to run each functionality in subsequent modules. Initially, navigate to configs\script_configurations\smoketest.json. In this file, input your username as the value for the key "test_username."

```json
{
  "function_to_call": "check_poetry_setup",
  "parameters": {
    "test_username": "your username goes here"
  }
}
```

Save the file and then execute the following command:

```bash
python main.py crop-classification\configs\scritp_configurations\smoketest.json
```

The output should resemble something like this:

```bash
All set!
Username, you are ready to go! :)
```

### 1. Feature extraction module

#### 1.1. Data structure

To execute the feature extraction module, you need to set up your data first. Refer to the [readme](./data/README.md) for a guide on how to structure your data.

#### 1.2. Setting up configs

Afterward, you'll need to configure two files. First, open your sentinel_channels.json. Choose the bands you want to use within a specific resolution and select the granule inside a specific route. You can add routes as needed. An example and the default configuration are as follows:

```json
{
  "resolutions": {
    "R10m": ["B02_10m", "B03_10m", "B04_10m", "B08_10m"],
    "R20m": ["B05_20m", "B06_20m", "B07_20m", "B8A_20m", "B11_20m", "B12_20m"],
    "R60m": ["B01_60m", "B09_60m"]
  },
  "routes": {
    "R036": [
      "T34TCQ",
      "T34TCR",
      "T34TCS",
      "T34TDQ",
      "T34TDR",
      "T34TDS",
      "T34TEQ",
      "T34TER"
    ],
    "R136": ["T34TDS", "T34TEQ", "T34TER"]
  }
}
```

Secondly, you need to configure pixel_feature_extraction.json. Add the path to your ground_truth file, date, and route of interest. Also, specify your output file where .geojson for each granule will be saved.

```json
{
  "function_to_call": "feature_extraction_for_single_date",
  "parameters": {
    "gdf_path": "crop-classification/data/ground_truths/ground_truths.geojson",
    "date": "20230613",
    "route": "R036",
    "output_path": "crop-classification/data/features"
  }
}
```

In the end, you just need to execute a single command to run your code.

```bash
python main.py crop-classification/configs/scritp_configurations/pixel_feature_extraction.json
```

Additionally, if you intend to run this module on multiple dates and routes, you can utilize the provided configuration file. Remember to add the corresponding route for each date.
```json
{
  "function_to_call": "feature_extraction_for_multiple_dates",
  "parameters": {
    "gdf_path": "crop-classification/data/ground_truths/ground_truths.geojson",
    "dates": ["20230613", "20230623"],
    "routes": ["R036", "R036"],
    "output_path": "crop-classification/data/features"
  }
}
```

Finally, you just need to execute a single command to run your code.

```bash
python main.py crop-classification/configs/scritp_configurations/multiple_pixel_feature_extraction.json
```

### 2. Training module

When running training on the data obtained through feature extraction, you need to set up some .json configurations first. For instance, if you intend to train a Random Forest model, navigate to random_forrest_model.json and define your preferences. You can modify and add model parameters, but refrain from altering the model_name and function_to_call attributes. Additionally, you can set up feature_paths (the folder where you extracted features from the original image) and the path where you want to save the model. To understand how features should be structured, refer to the [readme](./data/README.md).

#### 2.1. Random forrest config file

```json
{
  "function_to_call": "train_model",
  "parameters": {
    "model_name": "RF",
    "model_parameters": {
      "n_estimators": 100,
      "random_state": 42
    },
    "features_path": "crop-classification\\data\\features",
    "output_path": "path\\to\\your\\file_destination"
  }
}
```

To initiate the training of this model, execute the following command:

```bash
python main.py path/to/your/random_forrest_model.json
```

#### 2.2. Support vector machine config file

To initiate the training of this model, execute the following command:

```json
{
    "function_to_call": "train_model",
    "parameters": {
      "model_name": "SVM",
      "model_parameters": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale"
      },
      "train_features_path": "crop-classification\\data\\features_train",
      "output_path": "path\\to\\your\\file_destination",
      "model_directory_path": null
    }
  }
```

To commence the training of this model, execute the following command:

```bash
python main.py path/to/your/support_vector_machine_model.json
```

### 3. Testing modules

The module enables you to test your model on test data. Initially, set up your test_model.json configuration file. There are three paths that require modification: the path to your test dataset features, the path to your trained model, and the path where you want to generate your classification report.

```json
{
    "function_to_call": "test_model",
    "parameters": {
        "test_features_path": "crop-classification\\data\\features_test",
        "model_directory_path": "path\\to\\your\\model",
        "model_report_path":"path\\to\\your\\model_report"
      }
}
```


To run test of this model run:

```bash
python main.py path/to/your/test_model.json
```

### 4. Creating masked prediction

- The Crop Prediction Mask Generation module facilitates the utilization of a trained model to produce raw prediction masks of crops from Sentinel images. The module requires the setup of a configuration file named make_mask.json. This configuration file defines parameters such as dates, routes, and granules for which predictions are desired. Masks will be generated for all granules specified in the sentinel_configurations section of the configuration file.

Additionally, the user needs to provide paths to the Sentinel directory where local data is stored, as well as specify the output path, model path, projection type, and color map. It's crucial to ensure that the color map specified in the configuration file has the same number of colors as the model's output categories.

```json
{
    "function_to_call": "make_masks",
    "parameters": {
      "dates": ["20230623", "20230713"],
      "routes": ["R036", "R036"],
      "path_to_sentinel": "D:\\data",
      "output_path": "C:\\Users\\Polen\\Desktop",
      "model_path": "C:\\Users\\Polen\\Desktop\\RandomForest_2024_03_14_15_38",
      "projection": "EPSG:32634",
      "color_map": [[131, 205, 144], 
      [131, 198, 207], 
      [239, 222, 172], 
      [226, 176, 212], 
      [238, 168, 193], 
      [182, 171, 240], 
      [124, 104, 91], 
      [130, 78, 99], 
      [0, 0, 0]]
    }
}
```


To run mask maker run:

```bash
python main.py path/to/your/make_mask.json
```
