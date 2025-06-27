"""
neural network model module for vehicle price prediction
implements advanced mlp with multiple embeddings and multi-input architecture
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout, BatchNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, List
import warnings

warnings.filterwarnings('ignore')


class NeuralNetworkModel:
    """
    advanced neural network model for vehicle price prediction
    uses multi-input architecture with embeddings for high-cardinality features
    """

    def __init__(self, random_state: int = 42):
        """
        initialize neural network model

        args:
            random_state: random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.history = None
        self.is_trained = False

        # set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        # model architecture parameters
        self.embedding_dims = {
            'model': 16,  # model has 20k+ categories
            'region': 8  # region has 400+ categories
        }

    def prepare_data(self, nn_features: Dict, y: pd.Series,
                     test_size: float = 0.2) -> Tuple:
        """
        prepare data for neural network training

        args:
            nn_features: dictionary with neural network features
            y: target variable
            test_size: proportion of test data
        returns:
            tuple of train and test data
        """
        # get indices for splitting
        indices = range(len(y))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=self.random_state
        )

        # split numeric features
        X_numeric = nn_features['numeric']
        X_num_train = X_numeric.iloc[train_idx].values
        X_num_test = X_numeric.iloc[test_idx].values

        # split embedding features
        X_embed_train, X_embed_test = {}, {}
        for col, values in nn_features['embedding'].items():
            X_embed_train[col] = values[train_idx].reshape(-1, 1)
            X_embed_test[col] = values[test_idx].reshape(-1, 1)

        # split one-hot features
        X_onehot = nn_features['onehot']
        X_onehot_train = X_onehot.iloc[train_idx].values
        X_onehot_test = X_onehot.iloc[test_idx].values

        # split target
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values

        print(f"data prepared - train: {len(y_train)}, test: {len(y_test)}")

        return {
            'train': {
                'numeric': X_num_train,
                'embedding': X_embed_train,
                'onehot': X_onehot_train,
                'target': y_train
            },
            'test': {
                'numeric': X_num_test,
                'embedding': X_embed_test,
                'onehot': X_onehot_test,
                'target': y_test
            }
        }

    def build_model(self, nn_features: Dict, embedding_vocab_sizes: Dict) -> Model:
        """
        build multi-input neural network model

        args:
            nn_features: neural network features dictionary
            embedding_vocab_sizes: vocabulary sizes for embedding layers
        returns:
            compiled keras model
        """
        print("building multi-input neural network...")

        # 1. numeric input branch
        numeric_input = Input(shape=(nn_features['numeric'].shape[1],), name='numeric')
        numeric_dense = Dense(64, activation='relu')(numeric_input)
        numeric_dense = BatchNormalization()(numeric_dense)

        # 2. embedding input branches
        embedding_inputs = []
        embedding_outputs = []

        for col, vocab_size in embedding_vocab_sizes.items():
            # embedding input
            embed_input = Input(shape=(1,), name=f'{col}_input')
            # embedding layer with appropriate dimension
            embed_dim = self.embedding_dims.get(col, min(16, vocab_size // 4))
            embed_layer = Embedding(vocab_size, embed_dim, name=f'{col}_embed')(embed_input)
            embed_flat = Flatten()(embed_layer)

            embedding_inputs.append(embed_input)
            embedding_outputs.append(embed_flat)

            print(f"embedding {col}: vocab_size={vocab_size}, embed_dim={embed_dim}")

        # 3. one-hot input branch
        onehot_input = Input(shape=(nn_features['onehot'].shape[1],), name='onehot')
        onehot_dense = Dense(64, activation='relu')(onehot_input)
        onehot_dense = BatchNormalization()(onehot_dense)

        # 4. concatenate all branches
        all_inputs = [numeric_input] + embedding_inputs + [onehot_input]
        all_features = [numeric_dense] + embedding_outputs + [onehot_dense]

        concat = Concatenate()(all_features)

        # 5. deep network with residual connections
        x = Dense(256, activation='relu')(concat)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)

        # residual block
        x1 = Dense(128, activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.3)(x1)

        x2 = Dense(128, activation='relu')(x1)
        x2 = BatchNormalization()(x2)

        # skip connection (adjust dimensions)
        x_skip = Dense(128)(x)
        x_res = Add()([x2, x_skip])
        x_res = Dropout(0.3)(x_res)

        # final layers
        x_final = Dense(64, activation='relu')(x_res)
        x_final = BatchNormalization()(x_final)
        x_final = Dropout(0.2)(x_final)

        # output layer
        output = Dense(1, name='price')(x_final)

        # create model
        model = Model(inputs=all_inputs, outputs=output)

        # compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        print(f"model built with {model.count_params():,} parameters")
        return model

    def train_model(self, data: Dict, epochs: int = 50,
                    batch_size: int = 128, validation_split: float = 0.2) -> None:
        """
        train neural network model

        args:
            data: prepared data dictionary
            epochs: number of training epochs
            batch_size: batch size for training
            validation_split: validation data proportion
        """
        print("training neural network model...")

        # prepare training data
        train_data = data['train']
        train_inputs = [train_data['numeric']] + \
                       list(train_data['embedding'].values()) + \
                       [train_data['onehot']]

        # callbacks for training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # train model
        self.history = self.model.fit(
            train_inputs,
            train_data['target'],
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True
        print("neural network training completed!")

    def evaluate_model(self, data: Dict) -> Dict:
        """
        evaluate trained model on test data

        args:
            data: prepared data dictionary
        returns:
            dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("model must be trained before evaluation")

        # prepare test data
        test_data = data['test']
        test_inputs = [test_data['numeric']] + \
                      list(test_data['embedding'].values()) + \
                      [test_data['onehot']]

        # make predictions
        y_pred = self.model.predict(test_inputs).flatten()
        y_test = test_data['target']

        # calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'y_test': y_test
        }

        print(f"\n🚀 neural network model performance:")
        print(f"mae: ${mae:,.2f}")
        print(f"rmse: ${rmse:,.2f}")
        print(f"r²: {r2:.4f}")

        return metrics

    def visualize_training(self, save_path: str = "results/") -> None:
        """
        visualize training history

        args:
            save_path: path to save plots
        """
        if self.history is None:
            raise ValueError("model must be trained before visualization")

        # create results directory
        os.makedirs(save_path, exist_ok=True)

        # plot training history
        plt.figure(figsize=(12, 4))

        # loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='train loss')
        plt.plot(self.history.history['val_loss'], label='val loss')
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()

        # mae plot
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='train mae')
        plt.plot(self.history.history['val_mae'], label='val mae')
        plt.title('model mae')
        plt.xlabel('epoch')
        plt.ylabel('mae')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{save_path}/nn_training_history.png", dpi=300, bbox_inches='tight')
        plt.show()

    def visualize_results(self, metrics: Dict, save_path: str = "results/") -> None:
        """
        visualize model results

        args:
            metrics: evaluation metrics dictionary
            save_path: path to save plots
        """
        # create results directory
        os.makedirs(save_path, exist_ok=True)

        y_test = metrics['y_test']
        y_pred = metrics['predictions']

        # create plots
        plt.figure(figsize=(12, 4))

        # predicted vs actual
        plt.subplot(1, 2, 1)
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('actual price')
        plt.ylabel('predicted price')
        plt.title('predicted vs actual prices')

        # residual plot
        plt.subplot(1, 2, 2)
        residuals = y_test - y_pred
        plt.scatter(y_test, residuals, alpha=0.3)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('actual price')
        plt.ylabel('residuals')
        plt.title('residual plot')

        plt.tight_layout()
        plt.savefig(f"{save_path}/nn_results.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"plots saved to {save_path}")

    def save_model(self, filepath: str = "models/neural_network_model.h5") -> None:
        """
        save trained model

        args:
            filepath: path to save model
        """
        if not self.is_trained:
            raise ValueError("model must be trained before saving")

        # create models directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # save model
        self.model.save(filepath)
        print(f"model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        load model from file

        args:
            filepath: path to model file
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_trained = True
        print(f"model loaded from {filepath}")

    def train_and_evaluate(self, nn_features: Dict, y: pd.Series,
                           embedding_vocab_sizes: Dict) -> Dict:
        """
        complete training and evaluation pipeline

        args:
            nn_features: neural network features dictionary
            y: target variable
            embedding_vocab_sizes: vocabulary sizes for embeddings
        returns:
            evaluation metrics dictionary
        """
        print("starting neural network training and evaluation...")

        # prepare data
        data = self.prepare_data(nn_features, y)

        # build model
        self.model = self.build_model(nn_features, embedding_vocab_sizes)

        # train model
        self.train_model(data)

        # evaluate model
        metrics = self.evaluate_model(data)

        # visualize results
        self.visualize_training()
        self.visualize_results(metrics)

        # save model
        self.save_model()

        print("neural network pipeline completed!")
        return metrics


# example usage
if __name__ == "__main__":
    # test neural network model
    from preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer

    # load and preprocess data
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess("../data/vehicles.csv")

    # engineer features
    engineer = FeatureEngineer()
    features = engineer.engineer_features(df.drop('price', axis=1), df['price'])

    # get embedding vocabulary sizes
    embedding_vocab_sizes = {}
    for col, encoded_values in features['neural_network']['embedding'].items():
        embedding_vocab_sizes[col] = len(np.unique(encoded_values))

    # train neural network model
    nn_model = NeuralNetworkModel()
    metrics = nn_model.train_and_evaluate(
        features['neural_network'],
        features['target'],
        embedding_vocab_sizes
    )