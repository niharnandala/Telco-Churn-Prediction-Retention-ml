# src/preprocessing.py

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(df: pd.DataFrame, target_column: str):
    """
    Splits dataframe into features and target.
    """
    df = df.copy()

    X = df.drop(columns=[target_column], errors="ignore")
    y = df[target_column]

    return X, y


def get_column_types(X: pd.DataFrame):
    """
    Finds numeric and categorical columns.
    """
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return numeric_columns, categorical_columns


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds the preprocessing transformer.
    """
    numeric_columns, categorical_columns = get_column_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns)
        ]
    )

    return preprocessor