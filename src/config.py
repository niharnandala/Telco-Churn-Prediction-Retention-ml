# src/config.py

import os


# -----------------------------
# base path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -----------------------------
# data paths
# -----------------------------
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.xlsx")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data_processed", "final_dataset.parquet")


# -----------------------------
# model paths
# -----------------------------

MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_telco_model.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_telco_model.pkl")


# -----------------------------
# reports path
# -----------------------------
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


# -----------------------------
# target
# -----------------------------
TARGET_COLUMN = "Churn Value"


# -----------------------------
# columns removed based on project decisions
# -----------------------------
COLUMNS_TO_DROP = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Label",
    "Churn Score",
    "CLTV",
    "Churn Reason"
]


# -----------------------------
# service columns used in feature engineering
# -----------------------------
SERVICE_COLUMNS = [
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies"
]


# -----------------------------
# logistic regression settings
# -----------------------------
LOGISTIC_CONFIG = {
    "class_weight": "balanced",
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": 42
}


# -----------------------------
# xgboost settings
# -----------------------------
XGB_CONFIG = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 42,
    "eval_metric": "logloss"
}


# -----------------------------
# threshold settings
# -----------------------------
DEFAULT_THRESHOLD = 0.60
THRESHOLD_CANDIDATES = [0.30, 0.40, 0.50, 0.60, 0.70]


# -----------------------------
# business assumptions for retention simulation
# -----------------------------
RETENTION_COST_PER_CUSTOMER = 500
RETAINED_CUSTOMER_VALUE = 5000
RETENTION_SUCCESS_RATE = 0.35


# -----------------------------
# app display settings
# -----------------------------
LOW_RISK_MAX = 0.30
MEDIUM_RISK_MAX = 0.60