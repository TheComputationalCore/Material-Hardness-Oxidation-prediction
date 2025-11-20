"""
Train the hardness prediction model.
This script:
- Loads training data
- Splits into train/test sets
- Builds sklearn pipeline
- Evaluates metrics
- Saves model + metadata
"""

from __future__ import annotations

from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.pipelines import (
    build_hardness_pipeline,
    HARDNESS_FEATURES,
)
from src.models.utils import (
    load_csv,
    save_model,
    save_metadata,
)


DATA_PATH = "data/hardness.csv"
MODEL_PATH = "models/hardness_model.joblib"
META_PATH = "models/hardness_metadata.json"


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute regression metrics for evaluation."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred),
    }


def train_hardness_model() -> Dict[str, Any]:
    """
    Train the hardness model and return training metadata.
    Raises:
        FileNotFoundError: When CSV is missing.
        KeyError: If required columns are missing.
    """

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = load_csv(DATA_PATH)

    missing_features = [f for f in HARDNESS_FEATURES if f not in df.columns]
    if missing_features:
        raise KeyError(
            f"Dataset missing required columns: {missing_features}. "
            f"Dataset columns: {list(df.columns)}"
        )

    # -----------------------------
    # Train-test split
    # -----------------------------
    X = df[HARDNESS_FEATURES]
    y = df["Hardness"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # -----------------------------
    # Fit model
    # -----------------------------
    pipeline = build_hardness_pipeline()
    pipeline.fit(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print("\n=============================")
    print(" Hardness Model Evaluation")
    print("=============================")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # -----------------------------
    # Save pipeline + metadata
    # -----------------------------
    save_model(pipeline, MODEL_PATH)

    save_metadata(
        META_PATH,
        "Hardness Model",
        HARDNESS_FEATURES,
        metrics,
    )

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {META_PATH}\n")

    return {
        "model_path": MODEL_PATH,
        "meta_path": META_PATH,
        "metrics": metrics,
    }


if __name__ == "__main__":
    train_hardness_model()
