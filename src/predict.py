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
    """
    distance = abs(probability - threshold)

    if distance >= 0.30:
        return "High"
    elif distance >= 0.15:
        return "Medium"
    else:
        return "Low"


def get_logistic_explanation(model, df):
    """
    Creates a simple local explanation using logistic regression coefficients.

    This is not SHAP. It is a lightweight approximation based on:
    transformed feature values * learned coefficients.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    transformed = preprocessor.transform(df)
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]

    if hasattr(transformed, "toarray"):
        transformed_row = transformed.toarray()[0]
    else:
        transformed_row = transformed[0]

    contributions = transformed_row * coefficients

    explanation_df = pd.DataFrame({
        "feature": feature_names,
        "contribution": contributions
    })

    explanation_df["abs_contribution"] = explanation_df["contribution"].abs()

    top_positive = (
        explanation_df[explanation_df["contribution"] > 0]
        .sort_values(by="contribution", ascending=False)
        .head(3)[["feature", "contribution"]]
    )

    top_negative = (
        explanation_df[explanation_df["contribution"] < 0]
        .sort_values(by="contribution", ascending=True)
        .head(3)[["feature", "contribution"]]
    )

    return {
        "top_positive": top_positive.to_dict(orient="records"),
        "top_negative": top_negative.to_dict(orient="records")
    }


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

    explanation = None
    if model_name == "Logistic Regression":
        explanation = get_logistic_explanation(model, df)

    return {
        "probability": round(float(probability), 4),
        "prediction": prediction,
        "label": label,
        "risk_segment": risk_segment,
        "threshold": threshold,
        "confidence": confidence,
        "model_name": model_name,
        "explanation": explanation
    }