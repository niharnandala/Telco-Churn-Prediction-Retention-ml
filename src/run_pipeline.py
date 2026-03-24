# src/run_pipeline.py

import os

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from src.cleaning import load_raw_data, clean_data
from src.features import add_features
from src.models import build_model_pipeline
from src.preprocessing import split_features_target
from src.evaluation import (
    get_classification_metrics,
    create_threshold_table,
    compare_models,
    simulate_retention_strategy,
    simulate_retention_scenarios,
    save_metrics_json,
    save_threshold_table,
    save_model_comparison,
    save_retention_table,
    save_retention_scenarios
)
from src.config import (
    TARGET_COLUMN,
    PROCESSED_DATA_PATH,
    LOGISTIC_MODEL_PATH,
    XGB_MODEL_PATH,
    MODEL_DIR,
    DEFAULT_THRESHOLD,
    REPORTS_DIR
)


def make_output_folders():
    """
    Creates folders needed for saving outputs.
    """
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def prepare_data():
    """
    Loads raw data, applies cleaning, adds engineered features,
    and saves the final processed base dataset.
    """
    df = load_raw_data()
    df = clean_data(df)

    monthly_charge_median = df["Monthly Charges"].median() if "Monthly Charges" in df.columns else None

    df = add_features(df, monthly_charge_median=monthly_charge_median)

    df.to_parquet(PROCESSED_DATA_PATH, index=False)

    return df, monthly_charge_median


def train_and_evaluate_model(X_train, X_test, y_train, y_test, model_name):
    """
    Trains one model pipeline and returns:
    - trained pipeline
    - predicted probabilities
    - evaluation metrics
    """
    pipeline = build_model_pipeline(X_train, model_name=model_name)
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = get_classification_metrics(
        y_true=y_test,
        y_prob=y_prob,
        threshold=DEFAULT_THRESHOLD
    )

    return pipeline, y_prob, metrics


def save_feature_importance(trained_pipeline, model_name):
    """
    Saves feature importance or model coefficients to reports/.
    """
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    model = trained_pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if model_name == "logistic":
        values = model.coef_[0]
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "weight": values,
            "abs_weight": abs(values)
        }).sort_values(by="abs_weight", ascending=False)

        importance_df.to_csv(
            os.path.join(REPORTS_DIR, "feature_importance_logistic.csv"),
            index=False
        )

    elif model_name == "xgboost":
        values = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": values
        }).sort_values(by="importance", ascending=False)

        importance_df.to_csv(
            os.path.join(REPORTS_DIR, "feature_importance_xgboost.csv"),
            index=False
        )


def main():
    make_output_folders()

    print("Loading and preparing data...")
    df, monthly_charge_median = prepare_data()

    print("Splitting features and target...")
    X, y = split_features_target(df, TARGET_COLUMN)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training Logistic Regression...")
    logistic_pipeline, logistic_prob, logistic_metrics = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, model_name="logistic"
    )

    print("Training XGBoost...")
    xgb_pipeline, xgb_prob, xgb_metrics = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, model_name="xgboost"
    )

    print("Saving trained models...")
    joblib.dump(
        {
            "model": logistic_pipeline,
            "monthly_charge_median": monthly_charge_median
        },
        LOGISTIC_MODEL_PATH
    )

    joblib.dump(
        {
            "model": xgb_pipeline,
            "monthly_charge_median": monthly_charge_median
        },
        XGB_MODEL_PATH
    )

    print("Saving feature importance reports...")
    save_feature_importance(logistic_pipeline, model_name="logistic")
    save_feature_importance(xgb_pipeline, model_name="xgboost")

    print("Creating model comparison report...")
    model_results = {
        "logistic_regression": logistic_metrics,
        "xgboost": xgb_metrics
    }

    comparison_df = compare_models(model_results)
    save_model_comparison(comparison_df)

    print("Creating threshold report for logistic regression...")
    threshold_df = create_threshold_table(y_test, logistic_prob)
    save_threshold_table(threshold_df)

    print("Creating retention simulation report for logistic regression...")
    retention_df = simulate_retention_strategy(y_test, logistic_prob)
    save_retention_table(retention_df)

    print("Creating retention scenario analysis report...")
    scenario_df = simulate_retention_scenarios(y_test, logistic_prob)
    save_retention_scenarios(scenario_df)

    print("Saving final metrics...")
    final_metrics = {
        "final_model": "logistic_regression",
        "logistic_regression": logistic_metrics,
        "xgboost": xgb_metrics
    }
    save_metrics_json(final_metrics)

    print("Pipeline run completed.")
    print(f"Final deployed logistic model saved at: {LOGISTIC_MODEL_PATH}")
    print(f"Comparison xgboost model saved at: {XGB_MODEL_PATH}")


if __name__ == "__main__":
    main()