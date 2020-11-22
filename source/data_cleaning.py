import pandas as pd
import numpy as np
import os
from typing import List
from typing import Dict
#from pandas_profiling import ProfileReport

pd.options.display.max_rows = 20
pd.options.display.max_columns = None

"""DEFAULT VALUES SETUP"""
REAL_ESTATE_CSV_FILEPATH = os.path.dirname(os.getcwd()) + r"\data" + "\clean_dataset.csv"
CLEANED_CSV_FILEPATH = os.getcwd()+"\\outputs"+"\\df_after_cleaning.csv"

# for terrace and garden it will be double checked if their related boolean is True when area is 0
COLUMNS_NAN_REPLACE_WITH = {"terrace_area": 0, "garden_area": 0, "facades_number": 0}
# TBC
OUTLIERS_METHODS = ["fence_tukey_min", "fence_tukey_max", "5%", "95%"]
COLUMNS_OUTLIERS_IGNORE = ["terrace_area", "source", "garden_area"]
# during the profiling dominant values found in equipped kitchen (true, 83.7%), furnished (False, 96.3%), Open Fire (False, 93.5%)
# columns with a dominant value (inc. source and swimming pool) are excluded from duplicates check.
# related boolean columns (e.g. garden_has) are also excluded.
# region is excluded having already postcode
# building_state_agg is excluded since aggregation of different approaches from different sources.
# more info in previous project repository https://github.com/FrancescoMariottini/residential-real-estate-analysis/blob/main/README.md
COLUMNS_DUPLICATES_CHECK = ["postcode", "house_is", "property_subtype", "price", "rooms_number", "area"]
#only median aggregator was ketp in line with statbel approach and available results
COLUMNS_GROUPED_BY = {"facades_number": "property_subtype"}
GROUPBY_AGGREGATORS = [np.median] #[min,max,np.mean,np.median,len]
AGGREGATOR_COLUMNS = {min:"min",max:"max",np.mean:"mean",np.median:"median",len:"len"}


#Not used yet
REPORT_HTML_FILEPATH = os.getcwd() + "\\reports" + "\\df_before_cleaning.html"


def get_columns_with_nan(df: pd.DataFrame) -> List[str]:
    columns_with_nan = []
    for c in df.columns:
        c_na_count = df[c].isna().sum()
        if c_na_count > 0:
            columns_with_nan.append(c)
    return columns_with_nan

def describe_with_tukey_fences(df: pd.DataFrame,
                               percentiles: List[float] = [0.95, 0.94, 0.75, 0.5, 0.25, 0.06, 0.05]) -> pd.DataFrame:
    df_desc = df.describe(percentiles, include=np.number)
    df_index = df_desc.index.to_list()
    fence_tukey_min: List = [df_desc.loc["25%", c] - 1.5 * (df_desc.loc["75%", c] - df_desc.loc["25%", c]) for c in
                             df_desc.columns]
    fence_tukey_max: List = [df_desc.loc["75%", c] + 1.5 * (df_desc.loc["75%", c] - df_desc.loc["25%", c]) for c in
                             df_desc.columns]
    df_desc = df_desc.append(dict(zip(df_desc.columns, fence_tukey_min)), ignore_index=True)
    df_desc = df_desc.append(dict(zip(df_desc.columns, fence_tukey_max)), ignore_index=True)
    df_index.append('fence_tukey_min')
    df_index.append('fence_tukey_max')
    df_desc.index = df_index
    return df_desc


def get_outliers_index(df: pd.DataFrame, outliers_methods: List[str] = OUTLIERS_METHODS, columns: List[str] = None):
    df_desc: pd.DataFrame = describe_with_tukey_fences(df)
    if columns is None:
        columns = df.columns
    df_outliers = pd.DataFrame(columns=["column", "method", "count", "%", "index"])
    for c in columns:
        t_min, t_max, p95, p94, p06, p05 = df_desc.loc[
            ["fence_tukey_min", "fence_tukey_max", "95%", "94%", "6%", "5%"], c]
        for m in outliers_methods:
            # TBC elif (fence_tukey_max < p95) AND (p95 != p94):
            if m == "fence_tukey_min" or m == "5%":
                index = df[df[c] < df_desc.loc[m, c]].index
            elif m == "fence_tukey_max" or m == "95%":
                index = df[df[c] < df_desc.loc[m, c]].index
            df_outliers = df_outliers.append({"column": c, "method": m, "count": len(index),
                                              "%": round(len(index) / len(df), 2),
                                              "index": index}, ignore_index=True)
    return df_outliers

#QA how specify list of functions in typing ?
def add_grouped_parameter(df: pd.DataFrame, group_parameter: Dict[str, str] = COLUMNS_GROUPED_BY,
                          groupby_aggregators: List = GROUPBY_AGGREGATORS):
    for key, value in group_parameter.items():
        df_grp = df.groupbyd.groupby(value)[key].agg(groupby_aggregators)
        for aggregator in groupby_aggregators:
            column_name = f"{value}_{AGGREGATOR_COLUMNS[aggregator]}_{key}"
            def get_aggregated_value(x):
                return df_grp.loc[x, aggregator]
            df[column_name] = df[value].apply(get_aggregated_value())
    return df

class DataCleaning:
    def __init__(self,
                 real_estate_csv_filepath: str = REAL_ESTATE_CSV_FILEPATH,
                 columns_nan_replace_with: Dict[str, int] = COLUMNS_NAN_REPLACE_WITH,
                 columns_duplicates_check: List[str] = COLUMNS_DUPLICATES_CHECK,
                 columns_outliers_ignore: List[str] = COLUMNS_OUTLIERS_IGNORE,
                 outliers_methods: List[str] = OUTLIERS_METHODS,
                 cleaned_csv_path: str = CLEANED_CSV_FILEPATH,
                 report_html_filepath: str = REPORT_HTML_FILEPATH
                 ):
        self.df_0: pd.DataFrame = pd.read_csv(real_estate_csv_filepath)
        self.df_out: pd.DataFrame = self.df_0.copy(deep=True)
        self.columns_with_nan: List[str] = []
        self.index_removed_by_process: Dict[str, List] = {}
        self.outliers = pd.DataFrame(columns=["column", "method", "count", "%", "index"])

        self.columns_nan_replace_with = columns_nan_replace_with
        self.columns_duplicates_check = columns_duplicates_check
        self.columns_outliers_ignore = columns_outliers_ignore
        self.outliers_methods = outliers_methods
        self.cleaned_csv_path = cleaned_csv_path

        #not used yet
        self.report_html_filepath = report_html_filepath

    def fill_na(self, df_before: pd.DataFrame = None,
                columns_nan_replace_with: Dict[str, int] = None,
                inplace = True) -> pd.DataFrame:
        if df_before is None:
            df_before = self.df_out
        if columns_nan_replace_with is None:
            columns_nan_replace_with = self.columns_nan_replace_with
        df_out = df_before.fillna(columns_nan_replace_with)
        if inplace is True:
            self.df_out = df_out
        return df_out

    def drop_duplicates(self, df_before: pd.DataFrame = None, columns_duplicates_check: List = None,
                        inplace = True)  -> (pd.DataFrame, List):
        if df_before is None:
            df_before = self.df_out
        if columns_duplicates_check is None:
            columns_duplicates_check = self.columns_duplicates_check
        df_out = df_before.drop_duplicates(subset=columns_duplicates_check)
        index_dropped = df_before.index.difference(df_out).tolist()
        if inplace is True:
            self.df_out = df_out
            self.index_removed_by_process["duplicates_removed"]: List = index_dropped
        return df_out, index_dropped

    def get_outliers(self, df_before: pd.DataFrame = None, columns_outliers_ignore: List = None,
                     outliers_methods: List[str] = None, inplace = True) -> pd.DataFrame:
        if df_before is None:
            df_before = self.df_out
        if columns_outliers_ignore is None:
            columns_outliers_ignore = self.columns_outliers_ignore
        if outliers_methods is None:
            outliers_methods = self.outliers_methods
        columns = [c for c in df_before.columns not in columns_outliers_ignore]
        df_outliers = get_outliers_index(df_before, outliers_methods, columns)
        if inplace is True:
            self.outliers = df_outliers
        return df_outliers

    def drop_outliers(self, df_before: pd.DataFrame = None, columns_outliers_ignore: List = None,
                      outliers_methods: List[str] = None, inplace = True) -> (pd.DataFrame, List):
        if df_before is None:
            df_before = self.df_out
        if columns_outliers_ignore is None:
            columns_outliers_ignore = self.columns_outliers_ignore
        if outliers_methods is None:
            outliers_methods = self.outliers_methods
        df_outliers = self.get_outliers(df_before, columns_outliers_ignore, outliers_methods)
        index_dropped = []
        for index, row in df_outliers:
            count, index = df_outliers.loc[index, ["count", "index"]]
            if count > 1:
                for i in index:
                    if i not in index_dropped:
                        index_dropped.append(i)
        df_out = df_before[~df_before.index.isin(index_dropped)]
        if inplace is True:
            self.index_removed_by_process["outliers_removed"]: List = index_dropped
            self.df_out = df_out
        return df_out, index_dropped

    def get_cleaned_dataframe(self, df_before: pd.DataFrame = None, cleaned_csv_path: str = None):
        if df_before is None:
            df_out = self.df_0
            cleaned_csv_path = self.cleaned_csv_path
        elif df_before is not None:
            df_out = df_before.copy(deep=True)
        print(f"Initial dataset shape: {df_before.shape}")
        df_out = self.fill_na(df_before)
        df_out, index_dropped = self.drop_duplicates(df_out)
        print(f"{len(index_dropped)} dropped duplicates, resulting in dataset of shape {df_out.shape}")
        df_out, index_dropped = self.drop_outliers(df_out)
        print(f"{len(index_dropped)} dropped outliers, resulting in dataset of shape {df_out.shape}")
        if cleaned_csv_path is not None:
            df_out.to_csv(cleaned_csv_path)
        return df_out


dc = DataCleaning()
