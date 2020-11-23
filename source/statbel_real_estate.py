import pandas as pd
import numpy as np
import os
from typing import List
from typing import Dict

pd.options.display.max_rows = 20
pd.options.display.max_columns = None

REAL_ESTATE_CSV_FILEPATH = os.path.dirname(os.getcwd()) + r"\data" + "\clean_dataset.csv"
STATBEL_REAL_ESTATE_CSV_FILEPATH = os.path.dirname(os.getcwd()) + r"\data" + "\statbel_date_facades_price_median_and_selling_by_municipality.csv"
MUNICIPALITIES_CSV_FILEPATH = os.path.dirname(os.getcwd())+r"\data"+"\zipcode-belgium.csv"

class DataMerging:
    def __init__(self,
                 statbel_csv_filepath: str = STATBEL_REAL_ESTATE_CSV_FILEPATH,
                 municipalities_csv_filepath: str = MUNICIPALITIES_CSV_FILEPATH,
                 real_estate_csv_filepath: str = REAL_ESTATE_CSV_FILEPATH
                 ):

        real_estate_0 = pd.read_csv(real_estate_csv_filepath)  
        real_estate_out = real_estate_0.copy(deep=True)
        
        municipalities = pd.read_csv(municipalities_csv_filepath, header=None, usecols=[0, 1])
        municipalities_columns = ['postcode', 'municipality']  # 'longitude', 'latitude']
        municipalities.rename(columns=dict(zip(municipalities.columns,  municipalities_columns)), inplace=True)
  
        statbel = pd.read_csv(statbel_csv_filepath, usecols=[3, 4, 5, 6, 7])
        statbel_columns = ["municipality", "year", "building_type", "price_median", "sellings"]
        statbel.rename(columns=dict(zip(statbel.columns, statbel_columns)), inplace=True)
        statbel.loc[:, "pm_per_s"] = statbel.apply(lambda x: x["price_median"] * x["sellings"], axis=1)
        statbel_g = statbel.loc[:, ["municipality", "pm_per_s", "sellings"]].groupby(by="municipality").agg(sum)
        statbel_g["price_median_years"] = statbel_g["pm_per_s"] / statbel_g["sellings"]
        statbel_g["price_median_years"] = statbel_g["price_median_years"].transform(lambda x: round(x, 2))

        statbel_g = statbel_g.merge(municipalities, on='municipality', how='left') #, indicator=True)

        real_estate_out = real_estate_out.merge(statbel_g.loc[:, ["price_median_years", "postcode"]], on='postcode', how='left') # , indicator=True)


dm = DataMerging()