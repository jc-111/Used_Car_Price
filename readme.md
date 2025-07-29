# Used Car Price Prediction Package

A modular machine learning package for predicting used car prices using XGBoost and Neural Network models.  
Supports both training (with GPU) and deployment (CPU, with Streamlit UI and batch prediction).

---

## 📁 Project Structure

```
Used_Car_Price/
│
├── src/
│   ├── preprocessing.py         # Data cleaning and preprocessing
│   ├── feature_engineering.py   # Feature engineering for both models
│   ├── xgboost_model.py         # XGBoost model class
│   ├── neural_network.py        # Neural Network model class
│
├── main.py                      # Training pipeline (run on training server)
├── streamlit_app.py             # Streamlit UI for deployment (run on deploy server)
├── generate_fake_cars.py        # Script to generate fake test data
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files and folders to ignore in git
├── readme.md                    # This file
├── models/                      # Saved models (created after training)
└── test data/                   # Test data CSVs for deployment and validation
```

---

## 🚦 Server Roles

### 1. **Training Server (with GPU)**
- **Purpose:** Train models, evaluate, and save them for deployment.
- **Key Script:** `main.py`
- **Outputs:**  
  - `models/xgboost_model.joblib`  
  - `models/neural_network_model.h5`

### 2. **Deploy Server (CPU, Local Machine)**
- **Purpose:** Load trained models, predict prices for new data, provide UI and batch prediction.
- **Key Script:** `streamlit_app.py`
- **Test Data:** Test data (CSV) is **mandatory** and should be placed in the `test data/` folder for batch prediction and validation.
- **Test Data Generation:** Use `generate_fake_cars.py` if you need synthetic test data.

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone <your_repo_url>
   cd Used_Car_Price
   ```

2. **Create and activate a virtual environment (recommended):**
   ```sh
   python3 -m venv price_analyzer
   source price_analyzer/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

---

## 🏋️‍♂️ Training Workflow (Training Server)

1. **Prepare your training data:**  
   Place your raw CSV (e.g., `vehicles.csv`) in the `data/` directory.

2. **Run the training pipeline:**
   ```sh
   python main.py
   ```
   - This will preprocess data, engineer features, train both models, evaluate, and save them to `models/`.

3. **Transfer the `models/` directory to your deploy server.**

---

## 🚀 Deployment & Prediction (Deploy Server)

1. **Ensure the following are present on your deploy server:**
   - `models/` directory with trained models
   - `src/` directory with all modules
   - `streamlit_app.py`
   - `requirements.txt`
   - **Test data CSVs in the `test data/` folder** (mandatory for batch prediction)

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```sh
   streamlit run streamlit_app.py
   ```
   - Access the UI in your browser.
   - Enter car details for single prediction or upload a CSV from the `test data/` folder for batch prediction (test data required).

---

## 🧪 Generating Fake Test Data

To create synthetic data for testing or demo:

```sh
python generate_fake_cars.py
```
- This will generate:
  - `fake_cars_for_prediction.csv` (for batch prediction, no price column)
  - `fake_cars_for_validation.csv` (with price column for validation)

Move these files into the `test data/` folder for use in the deployment UI.

---

## 📦 Features

- **Modular codebase:** Clean separation of preprocessing, feature engineering, and modeling.
- **Supports both XGBoost and Neural Network models.**
- **Streamlit UI:** For easy local deployment and user-friendly predictions.
- **Batch prediction:** Upload a CSV from `test data/` and get predictions for all entries (test data required).
- **Fake data generation:** For testing and demonstration.

---

## 📝 Notes

- Do **not** retrain models on the deploy server; only use for inference.
- Make sure the input data columns match those expected by the models.
- For best results, use the same preprocessing and feature engineering pipeline for both training and inference.

---

## 📧 Contact

For questions or contributions, please open an issue or pull request on the repository.