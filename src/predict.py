# src/predict.py

import joblib
import pandas as pd

from src.config import LOGISTIC_MODEL_PATH, DEFAULT_THRESHOLD
from src.features import add_features


def load_model():
    saved_obj = joblib.load(LOGISTIC_MODEL_PATH)
    model = saved_obj["model"]
    monthly_charge_median = saved_obj["monthly_charge_median"]
    return model, monthly_charge_median


def predict_single(input_dict: dict, threshold=DEFAULT_THRESHOLD):
    model, monthly_charge_median = load_model()

    df = pd.DataFrame([input_dict])
    df = add_features(df, monthly_charge_median=monthly_charge_median)

    probability = model.predict_proba(df)[:, 1][0]
    prediction = int(probability >= threshold)
    label = "Churn" if prediction == 1 else "No Churn"

    if probability < 0.30:
        risk_segment = "Low Risk"
    elif probability < 0.60:
        risk_segment = "Moderate Risk"
    else:
        risk_segment = "High Risk"

    explanation = None
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

    explanation = {
        "top_positive": top_positive.to_dict(orient="records"),
        "top_negative": top_negative.to_dict(orient="records")
    }

    return {
        "probability": round(float(probability), 4),
        "prediction": prediction,
        "label": label,
        "risk_segment": risk_segment,
        "threshold": threshold,
        "model_name": "Logistic Regression",
        "explanation": explanation
    }