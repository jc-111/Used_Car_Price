import streamlit as st
import numpy as np
import pandas as pd
import joblib
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.xgboost_model import XGBoostModel
from src.neural_network import NeuralNetworkModel

# Load models once
@st.cache_resource
def load_models():
    xgb_model = XGBoostModel()
    xgb_model.load_model('models/xgboost_model.joblib')
    nn_model = NeuralNetworkModel()
    # Directly load Keras model with compile=False and assign to nn_model.model
    from tensorflow import keras
    nn_model.model = keras.models.load_model('models/neural_network_model.h5', compile=False)
    return xgb_model, nn_model

xgb_model, nn_model = load_models()

st.title("Used Car Price Prediction")

# Example: Collect user input for features
st.header("Enter Car Details")
year = st.number_input("Year", min_value=1980, max_value=2025, value=2015)
odometer = st.number_input("Odometer (miles)", min_value=0, max_value=500000, value=60000)
manufacturer = st.text_input("Manufacturer", "toyota")
model = st.text_input("Model", "camry")
region = st.text_input("Region", "los angeles")
fuel = st.selectbox("Fuel", ["gas", "diesel", "electric", "hybrid", "other"])
transmission = st.selectbox("Transmission", ["automatic", "manual", "other"])
drive = st.selectbox("Drive", ["fwd", "rwd", "4wd", "other"])
type_ = st.text_input("Type", "sedan")
paint_color = st.text_input("Paint Color", "white")
state = st.text_input("State", "ca")

# Collect into DataFrame for preprocessing
input_dict = {
    "year": [year],
    "odometer": [odometer],
    "manufacturer": [manufacturer],
    "model": [model],
    "region": [region],
    "fuel": [fuel],
    "transmission": [transmission],
    "drive": [drive],
    "type": [type_],
    "paint_color": [paint_color],
    "state": [state],
    # Add other required fields as needed
}
input_df = pd.DataFrame(input_dict)

# Add car_age feature
input_df['car_age'] = pd.Timestamp.now().year - input_df['year']

# Preprocess and engineer features
preprocessor = DataPreprocessor()
engineer = FeatureEngineer()
input_df = preprocessor.remove_columns(input_df)
input_df = preprocessor.convert_datetime(input_df)
input_df = preprocessor.handle_missing_values(input_df)
features = engineer.engineer_features(input_df, pd.Series([0]))  # dummy target

# Predict (single)
xgb_pred = xgb_model.model.predict(features['xgboost'])[0]

nn_features = features['neural_network']
onehot = nn_features['onehot'].values
expected_onehot_dim = 132  # set this to the expected dimension from training

# Pad onehot if needed
if onehot.shape[1] < expected_onehot_dim:
    pad_width = expected_onehot_dim - onehot.shape[1]
    onehot = np.pad(onehot, ((0, 0), (0, pad_width)), mode='constant')

nn_input = [
    nn_features['numeric'].values,
    nn_features['embedding']['model'].reshape(-1, 1),
    nn_features['embedding']['region'].reshape(-1, 1),
    onehot
]
nn_pred = nn_model.model.predict(nn_input)[0][0]

st.subheader("Predicted Price")
st.write(f"XGBoost: ${xgb_pred:,.2f}")
st.write(f"Neural Network: ${nn_pred:,.2f}")

def batch_predict(uploaded_file, xgb_model, nn_model):
    st.subheader("Batch Prediction")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", df.head())

            # Define expected column order
            expected_order = [
                'year', 'odometer', 'car_age', 'manufacturer', 'model', 'region',
                'fuel', 'transmission', 'drive', 'type', 'paint_color', 'state'
            ]

            # Reindex DataFrame to expected order (ignore missing columns)
            df = df.reindex(columns=expected_order)

            # If any expected columns are missing, raise an error
            missing_cols = [col for col in expected_order if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns in uploaded file: {missing_cols}")
                return

            # Add car_age feature
            df['car_age'] = pd.Timestamp.now().year - df['year']

            # Preprocess and engineer features
            preprocessor = DataPreprocessor()
            engineer = FeatureEngineer()
            df_proc = preprocessor.remove_columns(df)
            df_proc = preprocessor.convert_datetime(df_proc)
            df_proc = preprocessor.handle_missing_values(df_proc)
            features = engineer.engineer_features(df_proc, pd.Series([0]*len(df_proc)))  # dummy target

            # XGBoost predictions
            xgb_preds = xgb_model.model.predict(features['xgboost'])

            # Neural Network predictions
            nn_features = features['neural_network']
            onehot = nn_features['onehot'].values
            expected_onehot_dim = 132  # set this to the expected dimension from training

            if onehot.shape[1] < expected_onehot_dim:
                pad_width = expected_onehot_dim - onehot.shape[1]
                onehot = np.pad(onehot, ((0, 0), (0, pad_width)), mode='constant')

            nn_input = [
                nn_features['numeric'].values,
                nn_features['embedding']['model'].reshape(-1, 1),
                nn_features['embedding']['region'].reshape(-1, 1),
                onehot
            ]
            nn_preds = nn_model.model.predict(nn_input).flatten()

            # Results DataFrame: only for rows that survived preprocessing
            results = df.iloc[df_proc.index].copy()
            results['XGBoost_Prediction'] = xgb_preds
            results['NeuralNetwork_Prediction'] = nn_preds

            # Optionally, show dropped rows
            dropped_rows = set(df.index) - set(df_proc.index)
            if dropped_rows:
                st.warning(f"{len(dropped_rows)} rows were dropped during preprocessing due to missing or invalid data.")

            st.write("Prediction Results:", results)
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.header("Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Choose a CSV file with car data", type=["csv"])
batch_predict(uploaded_file, xgb_model, nn_model)

# # Load the fitted encoder
# fitted_encoder = joblib.load("models/onehot_encoder.joblib")
