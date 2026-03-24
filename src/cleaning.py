# src/cleaning.py

import pandas as pd

from src.config import DATA_PATH, COLUMNS_TO_DROP


def load_raw_data(file_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw Excel dataset.
    """
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    return df


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that were removed in the notebook.
    """
    df = df.copy()

    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes sure Total Charges is numeric.
    """
    df = df.copy()

    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full raw-data cleaning step.
    """
    df = df.copy()
    df = drop_unwanted_columns(df)
    df = fix_total_charges(df)
    return df