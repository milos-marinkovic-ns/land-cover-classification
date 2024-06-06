from typing import Any
import pandas as pd

class VegetationIndicesCalculator:

    def __init__(self, df: pd.DataFrame, date: str, route: str) -> None:
        """
        Moudle for calculating vegetation inidces

        Parameters:
        df: gpd.DataFrame - DataFrame for calculatin indices
        date: str - Date of DataFrame
        route: str - Route of DataFrame
        """
        
        self.df: pd.DataFrame = df
        self.date: str = date
        self.route: str = route
        self.coastal_aerosol_b1 = df[f'B01_{date}_{route}']
        self.blue_b2 = df[f'B02_{date}_{route}']
        self.green_b3 = df[f'B03_{date}_{route}']
        self.red_b4 = df[f'B04_{date}_{route}']
        self.red_edge_b5 = df[f'B05_{date}_{route}']
        self.red_edge_b6 = df[f'B06_{date}_{route}']
        self.red_edge_b7 = df[f'B07_{date}_{route}']
        self.nir_b8 = df[f'B08_{date}_{route}']
        self.narrow_nir_b8a = df[f'B8A_{date}_{route}']
        self.water_vapour_b9 = df[f'B09_{date}_{route}']
        self.swir_b11 = df[f'B11_{date}_{route}']
        self.swir_b12 = df[f'B12_{date}_{route}']

        self.G = 2.5
        self.C1 = 6.0
        self.C2 = 7.5
        self.L = 0.5
        self.gamma = 1

    def calculate_ndvi(self):
        self.df[f'ndvi_{self.date}_{self.route}'] = (self.nir_b8 - self.red_b4) / (self.nir_b8 + self.red_b4)

    def calculate_ndvire1(self):
        self.df[f'ndvire1_{self.date}_{self.route}'] = (self.nir_b8 - self.red_edge_b5) / (self.nir_b8 + self.red_edge_b5)
        
    def calculate_ndvire2(self):
        self.df[f'ndvire2_{self.date}_{self.route}'] = (self.nir_b8 - self.red_edge_b6) / (self.nir_b8 + self.red_edge_b6)

    def calculate_ndvire3(self):
        self.df[f'ndvire3_{self.date}_{self.route}'] = (self.nir_b8- self.red_edge_b7) / (self.nir_b8 + self.red_edge_b7)

    def calculate_evi(self):
        self.df[f'evi_{self.date}_{self.route}'] = self.G * (self.nir_b8 - self.red_b4) / (self.nir_b8 + self.C1 * self.red_b4 - self.C2 * self.blue_b2 + self.L)

    def calculate_evi2(self):
        self.df[f'evi2_{self.date}_{self.route}'] = (self.nir_b8 - self.red_b4) / self.G*(self.nir_b8 + 2.4 * self.red_b4 - self.blue_b2)

    def calculate_vari(self):
        self.df[f'vari_{self.date}_{self.route}'] = (self.green_b3 - self.red_b4) / (self.green_b3 + self.red_b4 - self.blue_b2)

    def calculate_savi(self):
        self.df[f'savi_{self.date}_{self.route}'] = (self.nir_b8 - self.red_b4) / (self.nir_b8 + self.red_b4 + self.L) * (1 + self.L)

    def calculate_arvi(self):
        self.df[f'arvi_{self.date}_{self.route}'] = (self.nir_b8 - self.red_b4 + self.gamma*(self.blue_b2 - self.red_b4)) / (self.nir_b8 + self.red_b4 - self.gamma*(self.blue_b2 - self.red_b4))

    def calculate_gavi(self):
        self.df[f'gavi_{self.date}_{self.route}'] = (self.nir_b8 - self.green_b3 + self.gamma*(self.blue_b2 - self.red_b4)) / (self.nir_b8 + self.green_b3 - self.gamma*(self.blue_b2 - self.red_b4))

    def calculate_vdvi(self):
        self.df[f'vdvi_{self.date}_{self.route}'] = (2 * self.green_b3 - self.red_b4 - self.blue_b2) / (2 * self.green_b3 + self.red_b4 + self.blue_b2)
 
    def calculate_ndwi(self):
        self.df[f'ndwi_{self.date}_{self.route}'] = (self.nir_b8 - self.swir_b11) / (self.nir_b8 + self.swir_b11)

    def calculate_ndwi2(self):
        self.df[f'ndwi2_{self.date}_{self.route}'] = (self.green_b3 - self.nir_b8) / (self.green_b3 + self.nir_b8)

    def calculate_nli(self):
        self.df[f'nli_{self.date}_{self.route}'] = (((self.nir_b8/(2 ** 16))) ** 2 - self.red_b4 / (2 ** 16)) / (((self.nir_b8/(2 ** 16))) ** 2 + self.red_b4 / (2 ** 16))

    def calculate_nli2(self):
        self.df[f'nli2_{self.date}_{self.route}'] = (((self.nir_b8/(2 ** 16))) ** 2 - self.swir_b11 / (2 ** 16)) / (((self.nir_b8/(2 ** 16))) ** 2 + self.swir_b11 / (2 ** 16))

    def calculate_mnli(self):
        self.df[f'mnli_{self.date}_{self.route}'] = (((self.nir_b8 / (2 ** 16)) ** 2 - self.red_b4 / (2 ** 16)) * (1 + self.L)) / ((((self.nir_b8 / (2 ** 16)) ** 2 + self.red_b4 / (2 ** 16)) + self.L))

    def calculate_mnli2(self):
        self.df[f'mnli2_{self.date}_{self.route}'] = (((self.nir_b8 / (2 ** 16)) ** 2 - self.swir_b11 / (2 ** 16)) * (1 + self.L)) / ((((self.nir_b8 / (2 ** 16)) ** 2 + self.swir_b11 / (2 ** 16)) + self.L))

    def calculate_ndmi(self):
        self.df[f'ndmi_{self.date}_{self.route}'] = (self.narrow_nir_b8a - (self.swir_b11 - self.swir_b12)) / (self.narrow_nir_b8a + (self.swir_b11 + self.swir_b12))

    def calculate_tg(self):
        self.df[f'tg_{self.date}_{self.route}'] = (self.green_b3 - 0.39 * self.red_b4 - 0.61 * self.blue_b2) / 13000

    def calculate_gli(self):
        self.df[f'gli_{self.date}_{self.route}'] = (2 * self.green_b3 - self.red_b4 - self.blue_b2) / (2 * self.green_b3 + self.red_b4 + self.blue_b2 + 1)

    def calculate_ExG(self):
        self.df[f'ExG_{self.date}_{self.route}'] = 2 * self.green_b3 - self.red_b4 - self.blue_b2

    def calculate_CIVE(self):
        self.df[f'CIVE_{self.date}_{self.route}'] = 0.441 * self.red_b4 - 0.811 * self.green_b3 + 0.385 * self.blue_b2 + 18.78745 * 13000

    
    def __call__(self) -> None:
        """
        Calculates vegetattion indices, and adds those indices inside provided DataFrame 
        """
        methods_to_call = [method for method in dir(self) if method.startswith("calculate_")]
        for method_name in methods_to_call:
            method = getattr(self, method_name)
            method()