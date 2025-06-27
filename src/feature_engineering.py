import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
   def __init__(self):

       # drop columns with high missing rate (>50%)
       self.high_missing_cols = ['cylinders', 'condition']

       self.numeric_cols = ['year', 'car_age', 'odometer']

       # embedding features (high cardinality)
       self.embedding_cols = ['model', 'region']  # model: thousands, region: 404

       # one-hot features (low-medium cardinality)
       self.onehot_cols = ['manufacturer', 'fuel', 'transmission', 'drive',
                          'type', 'paint_color', 'state']  # state: 51 unique

       self.keep_cols = ['drive', 'paint_color', 'type']

       self.encoders = {}
       self.scaler = StandardScaler()

   def drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       drop columns with excessive missing values

       args:
           df: input dataframe
       returns:
           dataframe with high missing columns removed
       """

       cols_to_drop = [col for col in self.high_missing_cols if col in df.columns]
       df = df.drop(columns=cols_to_drop)
       print(f"dropped {len(cols_to_drop)} high missing columns: {cols_to_drop}")
       return df

   def handle_remaining_missing(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       handle missing values in remaining columns

       args:
           df: input dataframe
       returns:
           dataframe with missing values handled
       """
       df = df.copy()

       # fill categorical missing with 'unknown'
       categorical_cols = self.embedding_cols + self.onehot_cols
       for col in categorical_cols:
           if col in df.columns:
               missing_count = df[col].isna().sum()
               if missing_count > 0:
                   df[col] = df[col].fillna('unknown')
                   print(f"filled {missing_count} missing values in {col}")

       numeric_missing = df[self.numeric_cols].isnull().sum().sum()
       if numeric_missing > 0:
           df = df.dropna(subset=self.numeric_cols)
           print(f"dropped {numeric_missing} rows with missing numeric values")

       return df

   def create_xgboost_features(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       create features for xgboost model using label encoding

       args:
           df: preprocessed dataframe
       returns:
           dataframe with label encoded features for xgboost
       """

       X_xgb = df.copy()

       non_feature_cols = ['id', 'posting_date', 'VIN']
       for col in non_feature_cols:
           if col in X_xgb.columns:
               X_xgb = X_xgb.drop(columns=[col])
               print(f"dropped non-feature column: {col}")

       categorical_cols = self.embedding_cols + self.onehot_cols

       # label encode all categorical features for xgboost
       for col in categorical_cols:
           if col in X_xgb.columns:
               le = LabelEncoder()
               X_xgb[f'{col}_encoded'] = le.fit_transform(X_xgb[col].astype(str))
               self.encoders[f'{col}_xgb'] = le
               X_xgb = X_xgb.drop(columns=[col])

       # ensure only numeric columns remain
       numeric_columns = X_xgb.select_dtypes(include=[np.number]).columns
       X_xgb = X_xgb[numeric_columns]

       print(f"final xgboost features: {X_xgb.shape[1]} numeric columns")
       print(f"columns: {list(X_xgb.columns)}")

       return X_xgb

   def create_neural_network_features(self, df: pd.DataFrame) -> Dict:
       """
       create features for neural network with multi-input architecture

       args:
           df: preprocessed dataframe
       returns:
           dictionary with different feature types for neural network
       """
       print("creating neural network features...")

       # 1. numeric features (scaled)
       X_numeric = df[self.numeric_cols].copy()
       X_numeric_scaled = self.scaler.fit_transform(X_numeric)
       X_numeric_scaled = pd.DataFrame(X_numeric_scaled, columns=self.numeric_cols)

       # 2. embedding features (label encoded)
       X_embedding = {}
       for col in self.embedding_cols:
           if col in df.columns:
               le = LabelEncoder()
               encoded = le.fit_transform(df[col].astype(str))
               X_embedding[col] = encoded
               self.encoders[f'{col}_nn'] = le
               print(f"encoded {col}: {len(le.classes_)} unique categories for embedding")

       # 3. one-hot features
       onehot_cols_available = [col for col in self.onehot_cols if col in df.columns]
       if onehot_cols_available:
           ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
           X_onehot = ohe.fit_transform(df[onehot_cols_available])
           X_onehot = pd.DataFrame(X_onehot, columns=ohe.get_feature_names_out(onehot_cols_available))
           self.encoders['onehot_nn'] = ohe
       else:
           X_onehot = pd.DataFrame()

       print(f"created features - numeric: {X_numeric_scaled.shape[1]}, "
             f"embedding: {len(X_embedding)}, one-hot: {X_onehot.shape[1]}")

       return {
           'numeric': X_numeric_scaled,
           'embedding': X_embedding,
           'onehot': X_onehot
       }

   def engineer_features(self, df: pd.DataFrame, y: pd.Series) -> Dict:
       """
       complete feature engineering pipeline

       args:
           df: preprocessed dataframe
           y: target variable
       returns:
           dictionary with features for both model types
       """

       # drop high missing columns
       df_clean = self.drop_high_missing_columns(df)

       df_clean = self.handle_remaining_missing(df_clean)

       # create xgboost features
       X_xgb = self.create_xgboost_features(df_clean)
       X_xgb_selected = X_xgb
       print(f"using all {X_xgb.shape[1]} features for xgboost (no selection needed)")

       # create neural network features
       nn_features = self.create_neural_network_features(df_clean)

       print("feature engineering completed!")
       print(f"xgboost features shape: {X_xgb_selected.shape}")
       print(f"neural network features - numeric: {nn_features['numeric'].shape}, "
             f"embedding: {len(nn_features['embedding'])}, one-hot: {nn_features['onehot'].shape}")

       return {
           'xgboost': X_xgb_selected,
           'neural_network': nn_features,
           'target': y
       }

if __name__ == "__main__":
    from preprocessing import DataPreprocessor

    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess("../data/vehicles.csv")

    engineer = FeatureEngineer()
    features = engineer.engineer_features(df.drop('price', axis=1), df['price'])