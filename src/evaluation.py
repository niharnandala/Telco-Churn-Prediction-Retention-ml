# src/evaluation.py

import json
import os

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.config import (
    REPORTS_DIR,
    DEFAULT_THRESHOLD,
    THRESHOLD_CANDIDATES,
    RETENTION_COST_PER_CUSTOMER,
    RETAINED_CUSTOMER_VALUE,
    RETENTION_SUCCESS_RATE
)


def make_reports_folder():
    """
    Creates the reports folder if it does not already exist.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)


def get_classification_metrics(y_true, y_prob, threshold=DEFAULT_THRESHOLD):
    """
    Calculates the main classification metrics for a given threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    return metrics


def save_metrics_json(metrics_dict, file_name="metrics.json"):
    """
    Saves a metrics dictionary as a JSON file inside reports/.
    """
    make_reports_folder()

    output_path = os.path.join(REPORTS_DIR, file_name)

    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)


def create_threshold_table(y_true, y_prob, thresholds=THRESHOLD_CANDIDATES):
    """
    Builds a table showing how the model behaves across different thresholds.
    """
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        row = {
            "threshold": threshold,
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
            "customers_flagged": int((y_pred == 1).sum()),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn)
        }

        rows.append(row)

    threshold_df = pd.DataFrame(rows)
    return threshold_df


def save_threshold_table(threshold_df, file_name="threshold_metrics.csv"):
    """
    Saves the threshold comparison table.
    """
    make_reports_folder()

    output_path = os.path.join(REPORTS_DIR, file_name)
    threshold_df.to_csv(output_path, index=False)


def compare_models(model_results):
    """
    Converts model metric dictionaries into a comparison dataframe.

    Example input:
    {
        "logistic": {...},
        "xgboost": {...}
    }
    """
    comparison_rows = []

    for model_name, metrics in model_results.items():
        row = {"model_name": model_name}
        row.update(metrics)
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values(by="roc_auc", ascending=False).reset_index(drop=True)

    return comparison_df


def save_model_comparison(comparison_df, file_name="model_comparison.csv"):
    """
    Saves the model comparison table.
    """
    make_reports_folder()

    output_path = os.path.join(REPORTS_DIR, file_name)
    comparison_df.to_csv(output_path, index=False)


def simulate_retention_strategy(
    y_true,
    y_prob,
    thresholds=THRESHOLD_CANDIDATES,
    retention_cost=RETENTION_COST_PER_CUSTOMER,
    retained_value=RETAINED_CUSTOMER_VALUE,
    retention_success_rate=RETENTION_SUCCESS_RATE
):
    """
    Simulates a retention campaign across different thresholds.

    Assumptions:
    - every customer predicted as churn gets targeted
    - targeting each customer has a cost
    - only a fraction of true churners are actually saved
    """
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        customers_targeted = int(tp + fp)
        total_campaign_cost = customers_targeted * retention_cost

        successful_retentions = tp * retention_success_rate
        gross_retained_value = successful_retentions * retained_value
        net_value = gross_retained_value - total_campaign_cost

        row = {
            "threshold": threshold,
            "customers_targeted": customers_targeted,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "retention_success_rate": retention_success_rate,
            "successful_retentions_est": round(successful_retentions, 2),
            "campaign_cost": round(total_campaign_cost, 2),
            "gross_retained_value": round(gross_retained_value, 2),
            "net_value": round(net_value, 2)
        }

        rows.append(row)

    retention_df = pd.DataFrame(rows)
    retention_df = retention_df.sort_values(by="net_value", ascending=False).reset_index(drop=True)

    return retention_df


def save_retention_table(retention_df, file_name="retention_policy_table.csv"):
    """
    Saves the retention strategy simulation table.
    """
    make_reports_folder()

    output_path = os.path.join(REPORTS_DIR, file_name)
    retention_df.to_csv(output_path, index=False)