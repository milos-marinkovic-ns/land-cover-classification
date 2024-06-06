# Data directory

In this directory you will need to put your raw data you want to preprocess using our modules.

## Instructions

### 1. Sentinel-2 images

If you want to use module for feature extraction from Sentinel-2 images you'll have to create new directory called sentinel. And there paste
standard Sentinel-2 files. You don't have to put sentinel directory in this data directory, but make sure to add path to your directory to wanted script_configuration file.

```bash
|---data
  ∟|---sentinel
     ∟|---S2A_MSIL2A_20230501T093031_N0509_R136_T34TDQ_20230501T124754.SAFE
      |---...
      |---...
```

Also you need to add ground truths. First create ground_truths folder inside data foler. Then in that folder drag and drop your shape ground_truths.geojson/.shp file.

```bash
|---data
  ∟|---sentinel
     ∟|---S2A_MSIL2A_20230501T093031_N0509_R136_T34TDQ_20230501T124754.SAFE
      |---...
      |---...
  ∟|---ground_truths
     ∟|---ground_truths.geojson
```

### 2. Extracted features

Features for training should be inside folder features, and for each date there should be a directory containg same number of granules for each date. For example something like this:

```bash
|---data
  ∟|---features
     ∟|---features_20230613_R036
      |---T34TCQ_20230613_R036_features.csv
      |---T34TCR_20230613_R036_features.csv
     ∟|---features_20230623_R036
      |---T34TCQ_20230623_R036_features.csv
      |---T34TCR_20230623_R036_features.csv
  ∟|---ground_truths
     ∟|---ground_truths.geojson
```
