"""
pipelines.py
Defines sklearn preprocessing and model pipelines for:
- Hardness prediction
- Oxidation rate prediction

Contains:
- Canonical feature lists
- ColumnTransformer preprocessing
- Fully reproducible pipelines
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# -------------------------------------------------------------------
# Canonical feature order (used across training + inference)
# -------------------------------------------------------------------

HARDNESS_FEATURES = ["Material", "Current", "Heat_Input", "Carbon", "Manganese"]

OXIDATION_FEATURES = [
    "Material",
    "Current",
    "Heat_Input",
    "Soaking_Time",
    "Carbon",
    "Manganese",
]

# -------------------------------------------------------------------
# Column groups
# -------------------------------------------------------------------

categorical_cols = ["Material"]

numeric_cols_hardness = ["Current", "Heat_Input", "Carbon", "Manganese"]
numeric_cols_oxidation = ["Current", "Heat_Input", "Soaking_Time", "Carbon", "Manganese"]

# -------------------------------------------------------------------
# Preprocessors
# -------------------------------------------------------------------

def make_preprocessor(numeric_features):
    """
    Create a ColumnTransformer handling:
    - OneHotEncoder for categorical data
    - StandardScaler for numeric features
    """

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=float,
                ),
                categorical_cols,
            ),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )


preprocessor_hardness = make_preprocessor(numeric_cols_hardness)
preprocessor_oxidation = make_preprocessor(numeric_cols_oxidation)

# -------------------------------------------------------------------
# Pipeline Builders
# -------------------------------------------------------------------

def build_hardness_pipeline() -> Pipeline:
    """
    Build the full sklearn pipeline for hardness prediction.
    Simple baseline model: Linear Regression.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor_hardness),
            ("model", LinearRegression()),
        ]
    )


def build_oxidation_pipeline() -> Pipeline:
    """
    Build the full sklearn pipeline for oxidation rate prediction.
    Uses RandomForestRegressor for non-linear interactions.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor_oxidation),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            ),
        ]
    )
