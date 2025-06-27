import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
from typing import Tuple, Dict


class XGBoostModel:
    """
    xgboost model for vehicle price prediction
    handles training, evaluation, and visualization
    """

    def __init__(self, random_state: int = 42):
        """
        initialize xgboost model

        args:
            random_state: random seed for reproducibility
        """

        self.random_state = random_state
        self.model = None
        self.is_trained = False

        self.default_params = {
            'n_estimators': 200,
            'learning_rate': 0.1,
            'max_depth': 6,
            'tree_method': 'hist',
            'device': 'cpu',  # could change to cuda
            'random_state': random_state
        }

    def check_gpu_availability(self) -> str:
        """
        check if gpu is available for training

        returns:
            device string ('cuda' or 'cpu')
        """

        try:
            test_model = XGBRegressor(device='cuda', n_estimators=1)
            print("gpu detected, using cuda acceleration")
            return 'cuda'
        except:
            print("gpu not available, using cpu")
            return 'cpu'

    def split_data(self, X: pd.DataFrame, y: pd.Series,
                   test_size: float = 0.2) -> Tuple:
        """
        split data into train and test sets

        args:
            X: feature matrix
            y: target variable
            test_size: proportion of test data
        returns:
            tuple of (X_train, X_test, y_train, y_test)
        """

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        print(f"data split - train: {X_train.shape[0]}, test: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def train_default_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        train xgboost model with default parameters

        args:
            X_train: training features
            y_train: training target
        """

        device = self.check_gpu_availability()
        self.default_params['device'] = device

        self.model = XGBRegressor(**self.default_params)
        self.model.fit(X_train, y_train)
        self.is_trained = True

        print("default model training completed!")

    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                              cv_folds: int = 3) -> Dict:
        """
        perform hyperparameter tuning using grid search

        args:
            X_train: training features
            y_train: training target
            cv_folds: number of cv folds
        returns:
            best parameters dictionary
        """

        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.15],
            'max_depth': [4, 6, 8],
        }

        # base model
        base_model = XGBRegressor(
            tree_method='hist',
            device=self.check_gpu_availability(),
            random_state=self.random_state
        )

        grid_search = GridSearchCV(
            base_model, param_grid,
            cv=cv_folds, scoring='neg_mean_absolute_error',
            n_jobs=-1, verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        self.is_trained = True

        print(f"best parameters: {grid_search.best_params_}")
        print(f"best cv mae: {-grid_search.best_score_:.2f}")

        return grid_search.best_params_

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        evaluate trained model on test data

        args:
            X_test: test features
            y_test: test target
        returns:
            dictionary with evaluation metrics
        """

        if not self.is_trained:
            raise ValueError("model must be trained before evaluation")

        # make predictions
        y_pred = self.model.predict(X_test)

        # calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }

        print(f"\nxgboost model performance:")
        print(f"mae: ${mae:,.2f}")
        print(f"rmse: ${rmse:,.2f}")
        print(f"r²: {r2:.4f}")

        return metrics

    def get_feature_importance(self, feature_names: list, top_n: int = 20) -> pd.DataFrame:
        """
        get feature importance from trained model

        args:
            feature_names: list of feature names
            top_n: number of top features to return
        returns:
            dataframe with feature importance
        """

        if not self.is_trained:
            raise ValueError("model must be trained before getting feature importance")

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df

    def visualize_results(self, metrics: Dict, importance_df: pd.DataFrame, save_path: str = "results/") -> None:
        """
        create visualizations for model results

        args:
            metrics: evaluation metrics dictionary
            importance_df: feature importance dataframe
            save_path: path to save plots
        """
        os.makedirs(save_path, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # feature importance plot
        axes[0].barh(importance_df['feature'], importance_df['importance'])
        axes[0].set_xlabel('importance')
        axes[0].set_title(f'top {len(importance_df)} feature importance (xgboost)')
        axes[0].invert_yaxis()

        # predicted vs actual plot
        y_test = metrics.get('y_test', [])
        y_pred = metrics['predictions']

        if len(y_test) > 0:
            axes[1].scatter(y_test, y_pred, alpha=0.3)
            axes[1].plot([y_test.min(), y_test.max()],
                         [y_test.min(), y_test.max()], 'r--')
            axes[1].set_xlabel('actual price')
            axes[1].set_ylabel('predicted price')
            axes[1].set_title('predicted vs actual prices')

            residuals = y_test - y_pred
            axes[2].scatter(y_test, residuals, alpha=0.3)
            axes[2].axhline(0, color='red', linestyle='--')
            axes[2].set_xlabel('actual price')
            axes[2].set_ylabel('residuals')
            axes[2].set_title('residual plot')

        plt.tight_layout()
        plt.savefig(f"{save_path}/xgboost_results.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"plots saved to {save_path}/xgboost_results.png")

    def save_model(self, filepath: str = "models/xgboost_model.joblib") -> None:
        """
        save trained model to file

        args:
            filepath: path to save model
        """

        if not self.is_trained:
            raise ValueError("model must be trained before saving")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # save model
        joblib.dump(self.model, filepath)
        print(f"model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        load model from file

        args:
            filepath: path to model file
        """

        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"model loaded from {filepath}")

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series,
                           use_tuning: bool = False) -> Dict:
        """
        complete training and evaluation pipeline

        args:
            X: feature matrix
            y: target variable
            use_tuning: whether to use hyperparameter tuning
        returns:
            evaluation metrics dictionary
        """

        # split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # train model
        if use_tuning:
            self.hyperparameter_tuning(X_train, y_train)
        else:
            self.train_default_model(X_train, y_train)

        # evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        metrics['y_test'] = y_test  # add for visualization

        # get feature importance
        importance_df = self.get_feature_importance(X.columns)

        # visualize results
        self.visualize_results(metrics, importance_df)

        self.save_model()

        return metrics

if __name__ == "__main__":

    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer

    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess("../data/vehicles.csv")

    # engineer features
    engineer = FeatureEngineer()
    features = engineer.engineer_features(df.drop('price', axis=1), df['price'])

    # train model
    xgb_model = XGBoostModel()
    metrics = xgb_model.train_and_evaluate(
        features['xgboost'],
        features['target'],
        use_tuning=False
    )