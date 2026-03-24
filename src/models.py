# src/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.config import LOGISTIC_CONFIG, XGB_CONFIG
from src.preprocessing import build_preprocessor


def get_logistic_model() -> LogisticRegression:
    """
    Creates the notebook-faithful Logistic Regression model.
    """
    return LogisticRegression(
        class_weight=LOGISTIC_CONFIG["class_weight"],
        solver=LOGISTIC_CONFIG["solver"],
        max_iter=LOGISTIC_CONFIG["max_iter"],
        random_state=LOGISTIC_CONFIG["random_state"]
    )


def get_xgb_model() -> XGBClassifier:
    """
    Creates the comparison XGBoost model.
    """
    return XGBClassifier(
        n_estimators=XGB_CONFIG["n_estimators"],
        max_depth=XGB_CONFIG["max_depth"],
        learning_rate=XGB_CONFIG["learning_rate"],
        subsample=XGB_CONFIG["subsample"],
        colsample_bytree=XGB_CONFIG["colsample_bytree"],
        random_state=XGB_CONFIG["random_state"],
        eval_metric=XGB_CONFIG["eval_metric"]
    )


def build_model_pipeline(X, model_name: str = "logistic"):
    """
    Builds a full pipeline using the selected model.
    """
    preprocessor = build_preprocessor(X)

    if model_name == "logistic":
        model = get_logistic_model()
    elif model_name == "xgboost":
        model = get_xgb_model()
    else:
        raise ValueError("model_name must be either 'logistic' or 'xgboost'")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline