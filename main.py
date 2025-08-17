import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.xgboost_model import XGBoostModel
from src.neural_network import NeuralNetworkModel
from src.baseline_model import BaselineLinearModel


def create_directories():
   """create necessary directories for results and models"""
   directories = ['results', 'models']
   for directory in directories:
       os.makedirs(directory, exist_ok=True)
       print(f"created directory: {directory}")

def load_and_prepare_data():
   """
   load raw data and perform preprocessing and feature engineering

   returns:
       dictionary with processed features and target
   """
   print("STEP 1: Data Preprocessing")
   print("=" * 60)

   preprocessor = DataPreprocessor()

   df_clean = preprocessor.preprocess("data/vehicles.csv")

   print("STEP 2: Feature Engineering")
   print("=" * 60)

   engineer = FeatureEngineer()

   X = df_clean.drop('price', axis=1)
   y = df_clean['price']

   features = engineer.engineer_features(X, y)

   return features

def train_xgboost_model(features):
   """
   train and evaluate xgboost model

   args:
       features: processed features dictionary
   returns:
       xgboost metrics dictionary
   """

   print("STEP 3: XGBoost Model Training")
   print("=" * 60)

   # initialize xgboost model
   xgb_model = XGBoostModel(random_state=42)

   # train and evaluate
   xgb_metrics = xgb_model.train_and_evaluate(
       features['xgboost'],
       features['target'],
       use_tuning=True
   )
   print("XGBoost tuning artifacts: results/xgb_best_params.json, results/xgb_gridsearch_results.csv")
   return xgb_metrics, xgb_model

def train_neural_network_model(features):
   """
   train and evaluate neural network model

   args:
       features: processed features dictionary
   returns:
       neural network metrics dictionary
   """

   print("STEP 4: Neural Network Model Training")
   print("=" * 60)

   embedding_vocab_sizes = {}
   for col, encoded_values in features['neural_network']['embedding'].items():
       embedding_vocab_sizes[col] = len(np.unique(encoded_values))

   # initialize neural network model
   nn_model = NeuralNetworkModel(random_state=42)

   # train and evaluate
   nn_metrics = nn_model.train_and_evaluate(
       features['neural_network'],
       features['target'],
       embedding_vocab_sizes
   )

   return nn_metrics, nn_model

def compare_models(xgb_metrics, nn_metrics):
   """
   compare performance of both models

   args:
       xgb_metrics: xgboost evaluation metrics
       nn_metrics: neural network evaluation metrics
   """

   print("STEP 5: Comparing Models")
   print("=" * 60)

   # create comparison table
   comparison_data = {
       'Model': ['XGBoost', 'Neural Network'],
       'MAE ($)': [f"{xgb_metrics['mae']:,.2f}", f"{nn_metrics['mae']:,.2f}"],
       'RMSE ($)': [f"{xgb_metrics['rmse']:,.2f}", f"{nn_metrics['rmse']:,.2f}"],
       'R²': [f"{xgb_metrics['r2']:.4f}", f"{nn_metrics['r2']:.4f}"]
   }

   comparison_df = pd.DataFrame(comparison_data)
   print("\nModel Performance Comparison:")
   print(comparison_df.to_string(index=False))

   # determine best model
   if nn_metrics['r2'] > xgb_metrics['r2']:
       best_model = 'Neural Network'
       improvement = ((nn_metrics['r2'] - xgb_metrics['r2']) / xgb_metrics['r2']) * 100
   else:
       best_model = 'XGBoost'
       improvement = ((xgb_metrics['r2'] - nn_metrics['r2']) / nn_metrics['r2']) * 100

   print(f"\nBest Model: {best_model}")
   print(f"R² Improvement: {improvement:.2f}%")

   create_comparison_plots(xgb_metrics, nn_metrics)

   comparison_df.to_csv('results/model_comparison.csv', index=False)

def create_comparison_plots(xgb_metrics, nn_metrics):
   """
   create comparison plots for both models

   args:
       xgb_metrics: xgboost evaluation metrics
       nn_metrics: neural network evaluation metrics
   """

   fig, axes = plt.subplots(2, 2, figsize=(15, 12))

   # metrics comparison
   models = ['XGBoost', 'Neural Network']
   mae_values = [xgb_metrics['mae'], nn_metrics['mae']]
   rmse_values = [xgb_metrics['rmse'], nn_metrics['rmse']]
   r2_values = [xgb_metrics['r2'], nn_metrics['r2']]

   # mae comparison
   axes[0, 0].bar(models, mae_values, color=['#1f77b4', '#ff7f0e'])
   axes[0, 0].set_title('Mean Absolute Error (MAE)')
   axes[0, 0].set_ylabel('MAE ($)')
   for i, v in enumerate(mae_values):
       axes[0, 0].text(i, v + 50, f'${v:,.0f}', ha='center', va='bottom')

   # rmse comparison
   axes[0, 1].bar(models, rmse_values, color=['#1f77b4', '#ff7f0e'])
   axes[0, 1].set_title('Root Mean Square Error (RMSE)')
   axes[0, 1].set_ylabel('RMSE ($)')
   for i, v in enumerate(rmse_values):
       axes[0, 1].text(i, v + 50, f'${v:,.0f}', ha='center', va='bottom')

   # r2 comparison
   axes[1, 0].bar(models, r2_values, color=['#1f77b4', '#ff7f0e'])
   axes[1, 0].set_title('R² Score')
   axes[1, 0].set_ylabel('R²')
   axes[1, 0].set_ylim(0.8, 1.0)
   for i, v in enumerate(r2_values):
       axes[1, 0].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')

   # prediction scatter comparison
   if 'predictions' in xgb_metrics and 'y_test' in xgb_metrics:
       y_test_xgb = xgb_metrics['y_test']
       y_pred_xgb = xgb_metrics['predictions']

       axes[1, 1].scatter(y_test_xgb, y_pred_xgb, alpha=0.3, label='XGBoost', s=1)

       if 'predictions' in nn_metrics and 'y_test' in nn_metrics:
           y_test_nn = nn_metrics['y_test']
           y_pred_nn = nn_metrics['predictions']
           axes[1, 1].scatter(y_test_nn, y_pred_nn, alpha=0.3, label='Neural Network', s=1)

       # perfect prediction line
       min_val = min(y_test_xgb.min(), y_test_nn.min() if 'y_test' in nn_metrics else y_test_xgb.min())
       max_val = max(y_test_xgb.max(), y_test_nn.max() if 'y_test' in nn_metrics else y_test_xgb.max())
       axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

       axes[1, 1].set_xlabel('Actual Price ($)')
       axes[1, 1].set_ylabel('Predicted Price ($)')
       axes[1, 1].set_title('Prediction Comparison')
       axes[1, 1].legend()
   else:
       axes[1, 1].text(0.5, 0.5, 'Prediction data not available',
                      ha='center', va='center', transform=axes[1, 1].transAxes)
       axes[1, 1].set_title('Prediction Comparison')

   plt.tight_layout()
   plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')

def main():
    try:
        create_directories()
        features = load_and_prepare_data()

        # Baseline: Linear Regression
        baseline = BaselineLinearModel(random_state=42, model_name="LinearRegression")
        baseline_metrics = baseline.train_and_evaluate(
            features['neural_network'],
            features['target']
        )
        print("\nBaseline (Linear Regression) performance:")
        print(f"MAE: ${baseline_metrics['mae']:,.2f}")
        print(f"RMSE: ${baseline_metrics['rmse']:,.2f}")
        print(f"R²: {baseline_metrics['r2']:.4f}")

        # xgboost model
        xgb_metrics, xgb_model = train_xgboost_model(features)
        xgb_model.save_model('models/xgboost_model.joblib')  # Save XGBoost

        # neural network model
        nn_metrics, nn_model = train_neural_network_model(features)
        nn_model.save_model('models/neural_network_model.h5')  # Save NN

        # compare models
        compare_models(xgb_metrics, nn_metrics)

    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
   main()
