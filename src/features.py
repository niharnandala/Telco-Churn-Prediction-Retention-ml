# src/features.py

import numpy as np
import pandas as pd

from src.config import SERVICE_COLUMNS


def add_features(df: pd.DataFrame, monthly_charge_median: float = None) -> pd.DataFrame:
    """
    Adds the same engineered features used in the notebook.

    monthly_charge_median can be passed from training so that
    HighRisk_Combo stays consistent in inference too.
    """
    df = df.copy()

    available_service_columns = [col for col in SERVICE_COLUMNS if col in df.columns]

    if available_service_columns:
        df["num_services"] = (df[available_service_columns] == "Yes").sum(axis=1)
    else:
        df["num_services"] = 0

    if "Tenure Months" in df.columns:
        df["Tenure_group"] = pd.cut(
            df["Tenure Months"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
            include_lowest=True
        )
    else:
        df["Tenure_group"] = np.nan

    if "Total Charges" in df.columns and "Tenure Months" in df.columns:
        safe_tenure = df["Tenure Months"].replace(0, np.nan)
        df["AvgCharges"] = df["Total Charges"] / safe_tenure
        df["AvgCharges"] = df["AvgCharges"].replace([np.inf, -np.inf], np.nan)
        df["AvgCharges"] = df["AvgCharges"].fillna(0)
    else:
        df["AvgCharges"] = 0

    if "Contract" in df.columns and "Monthly Charges" in df.columns:
        if monthly_charge_median is None:
            monthly_charge_median = df["Monthly Charges"].median()

        df["HighRisk_Combo"] = (
            (df["Contract"] == "Month-to-month") &
            (df["Monthly Charges"] > monthly_charge_median)
        ).astype(int)
    else:
        df["HighRisk_Combo"] = 0

    if available_service_columns:
        df["ServiceIntensity"] = df["num_services"] / len(SERVICE_COLUMNS)
    else:
        df["ServiceIntensity"] = 0

    return df