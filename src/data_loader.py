# src/data_loader.py

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the Telco churn dataset from an Excel file.

    I kept the raw loading here so the training file
    doesn't get messy with data cleaning steps.
    """
    df = pd.read_excel(file_path)

    # removing extra spaces from column names just in case
    df.columns = df.columns.str.strip()

    # making sure Total Charges is numeric
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    return df


def prepare_target(df: pd.DataFrame, target_col: str = "Churn Value") -> pd.DataFrame:
    """
    Makes sure the target column is numeric.

    In this dataset:
    0 = customer stayed
    1 = customer churned
    """
    df = df.copy()

    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    return df