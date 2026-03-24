# src/predict.py

import joblib
import pandas as pd

from src.config import LOGISTIC_MODEL_PATH, XGB_MODEL_PATH, DEFAULT_THRESHOLD
from src.features import add_features


MODEL_PATHS = {
    "Logistic Regression": LOGISTIC_MODEL_PATH,
    "XGBoost": XGB_MODEL_PATH
}


def load_model(model_name="Logistic Regression"):
    """
    Loads the saved model and related artifacts.
    """
    model_path = MODEL_PATHS[model_name]
    saved_obj = joblib.load(model_path)

    model = saved_obj["model"]
    monthly_charge_median = saved_obj["monthly_charge_median"]

    return model, monthly_charge_median


def get_confidence(probability, threshold):
    """
    Simple confidence score based on distance from threshold.

    This is not calibrated confidence.
    It is just a rough decision-confidence signal.
    """
    distance = abs(probability - threshold)

    if distance >= 0.30:
        return "High"
    elif distance >= 0.15:
        return "Medium"
    else:
        return "Low"


def predict_single(input_dict: dict, model_name="Logistic Regression", threshold=DEFAULT_THRESHOLD):
    """
    Takes one customer input and returns prediction details.
    """
    model, monthly_charge_median = load_model(model_name=model_name)

    df = pd.DataFrame([input_dict])
    df = add_features(df, monthly_charge_median=monthly_charge_median)

    probability = model.predict_proba(df)[:, 1][0]
    prediction = int(probability >= threshold)
    label = "Churn" if prediction == 1 else "No Churn"

    if probability < 0.30:
        risk_segment = "Low Risk"
    elif probability < 0.60:
        risk_segment = "Medium Risk"
    else:
        risk_segment = "High Risk"

    confidence = get_confidence(probability, threshold)

    return {
        "probability": round(float(probability), 4),
        "prediction": prediction,
        "label": label,
        "risk_segment": risk_segment,
        "threshold": threshold,
        "confidence": confidence,
        "model_name": model_name
    }