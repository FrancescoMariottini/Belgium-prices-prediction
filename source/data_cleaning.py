import pandas as pd
import numpy as np
import os
from typing import List

SOURCE_PATH = os.path.dirname(os.getcwd())+r"\data"+"\clean_dataset.csv"
COLUMNS_NAN_REPLACE = ["terrace_area", "garden_area", "land_surface", "price"]
COLUMNS_OUTLIERS_IGNORE = ["terrace_area", "source", "garden_area"]

def get_dataset_cleaned(filepath: str = SOURCE_PATH,
                        columns_nan_to_replace: List[str] = COLUMNS_NAN_REPLACE,
                        columns_outliers_ignore: List[str] = COLUMNS_OUTLIERS_IGNORE):
    df = pd.read_csv(filepath)

    df_cleaned = df.copy(deep=True)

    df_with_facades = df_cleaned[df_cleaned.facades_number > 0]

    property_subtype_facades = df_with_facades.groupby('property_subtype')['facades_number'].agg([min, max, np.mean, np.median, len])

"""
    print(property_subtype_facades.sort_values(by="median", ascending=False).head(5))
    print(property_subtype_facades.sort_values(by="mean", ascending=False).head(5))

    def get_property_subtype_facades(x):
        property_subtype_facades_median = property_subtype_facades.loc[x, "median"]
        return property_subtype_facades_median

    df_cleaned["property_subtype_median_facades"] = df_cleaned["property_subtype"].apply(get_property_subtype_facades)

#df_cleaned["property_subtype_median_facades"] = df_cleaned["property_subtype"].apply(get_property_subtype_facades)
"""

get_dataset_cleaned()