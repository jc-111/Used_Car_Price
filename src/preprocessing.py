import pandas as pd
import numpy as np
from typing import Tuple

class DataPreprocessor:
   def __init__(self):
       self.columns_to_drop = [
           'url', 'region_url', 'vin', 'image_url', 'description',
           'county', 'size', 'lat', 'long', 'title_status'
       ]
       self.fill_unknown_cols = [
           'condition', 'cylinders', 'fuel', 'drive',
           'transmission', 'paint_color', 'type', 'region', 'state'
       ]

   def load_data(self, file_path: str) -> pd.DataFrame:
       """
       load csv data from file path

       args:
           file_path: path to csv file
       returns:
           dataframe with raw data
       """
       print(f"loading data from {file_path}")
       df = pd.read_csv(file_path)
       print(f"loaded {df.shape[0]} rows, {df.shape[1]} columns")
       return df

   def remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       remove unnecessary columns for modeling

       args:
           df: input dataframe
       returns:
           dataframe with columns removed
       """

       # only drop columns that exist
       cols_to_drop = [col for col in self.columns_to_drop if col in df.columns]
       df = df.drop(columns=cols_to_drop, errors='ignore')
       print(f"removed {len(cols_to_drop)} columns")
       return df

   def convert_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       convert posting_date to datetime and create car_age feature

       args:
           df: input dataframe
       returns:
           dataframe with datetime conversion and car_age column
       """

       if 'posting_date' in df.columns:
           df['posting_date'] = pd.to_datetime(df['posting_date'], errors='coerce', utc=True)
           df['car_age'] = df['posting_date'].dt.year - df['year']
           print("converted posting_date and created car_age column")
       return df

   def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       remove extreme outliers based on business logic

       args:
           df: input dataframe
       returns:
           dataframe with outliers removed
       """
       original_size = len(df)

       # filter price range
       df = df[(df['price'] >= 1000) & (df['price'] <= 80000)]

       # filter car age
       if 'car_age' in df.columns:
           df = df[df['car_age'].between(0, 30, inclusive='both')]

       # filter odometer
       if 'odometer' in df.columns:
           df = df[df['odometer'] <= 1000000]

       removed = original_size - len(df)
       print(f"removed {removed} outlier rows ({removed/original_size*100:.1f}%)")
       return df

   def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       handle missing values - drop critical missing, fill others

       args:
           df: input dataframe
       returns:
           dataframe with missing values handled
       """
       # drop rows with missing critical values

       df = df.copy()

       critical_cols = ['price', 'year', 'manufacturer', 'model', 'odometer']
       critical_cols = [col for col in critical_cols if col in df.columns]

       original_size = len(df)
       df = df.dropna(subset=critical_cols)
       dropped = original_size - len(df)
       print(f"dropped {dropped} rows with missing critical values")

       # fill categorical missing with 'unknown'
       for col in self.fill_unknown_cols:
           if col in df.columns:
               missing_count = df[col].isna().sum()
               if missing_count > 0:
                   df[col] = df[col].fillna('unknown')
                   print(f"filled {missing_count} missing values in {col}")

       return df

   def get_data_summary(self, df: pd.DataFrame) -> None:
       """
       print summary statistics of processed data

       args:
           df: processed dataframe
       """

       print(f"final shape: {df.shape}")
       print(f"price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")

       if 'car_age' in df.columns:
           print(f"car age range: {df['car_age'].min():.0f} - {df['car_age'].max():.0f} years")

       print(f"missing values: {df.isnull().sum().sum()}")

   def preprocess(self, file_path: str) -> pd.DataFrame:
       """
       complete preprocessing pipeline

       args:
           file_path: path to raw csv file
       returns:
           cleaned and preprocessed dataframe
       """

       df = self.load_data(file_path)

       df = self.remove_columns(df)
       df = self.convert_datetime(df)
       df = self.remove_outliers(df)
       df = self.handle_missing_values(df)

       # reset index
       df = df.reset_index(drop=True)

       self.get_data_summary(df)

       print("preprocessing completed!")
       return df

if __name__ == "__main__":

   preprocessor = DataPreprocessor()
   df_clean = preprocessor.preprocess("../data/vehicles.csv")