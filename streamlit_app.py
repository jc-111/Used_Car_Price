import streamlit as st
import numpy as np
import pandas as pd
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
    nn_model.load_model('models/neural_network_model.h5')
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

# Preprocess and engineer features
preprocessor = DataPreprocessor()
engineer = FeatureEngineer()
input_df = preprocessor.remove_columns(input_df)
input_df = preprocessor.convert_datetime(input_df)
input_df = preprocessor.handle_missing_values(input_df)
features = engineer.engineer_features(input_df, pd.Series([0]))  # dummy target

# Predict
xgb_pred = xgb_model.model.predict(features['xgboost'])[0]
embedding_vocab_sizes = {col: len(np.unique(vals)) for col, vals in features['neural_network']['embedding'].items()}
nn_pred = nn_model.model.predict(features['neural_network'])[0][0]

st.subheader("Predicted Price")
st.write(f"XGBoost: ${xgb_pred:,.2f}")
st.write(f"Neural Network: ${nn_pred:,.2f}")

def batch_predict(uploaded_file, xgb_model, nn_model):
    st.subheader("Batch Prediction")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:", df.head())

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
            nn_preds = nn_model.model.predict(features['neural_network']).flatten()

            # Results DataFrame
            results = df.copy()
            results['XGBoost_Prediction'] = xgb_preds
            results['NeuralNetwork_Prediction'] = nn_preds

            st.write("Prediction Results:", results)
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")

st.header("Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Choose a CSV file with car data", type=["csv"])
batch_predict(uploaded_file, xgb_model, nn_model)