"""
regression_pipeline.py
=======================
Modular regression pipeline with preprocessing, feature engineering, 
cross-validation, and evaluvation.

Engineering decisions:
- All steps wrapped in sklearn Pipeline -> no data leakage during CV
- Config dataclass keeps hyperparameters out of bussiness logic
- Logging over print statments
- Type hints throught
- design to be imported as a module, not just run as a script

Usage:
    python foundations/ml-experiments/regression_pipeline.py
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler

warnings. filterwarings("igonre")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Central configuration for the regression pipeline.

    Keeping hyperparameters here instead of scattered through the code
    makes experiments reproducible and easy to modify.
    """
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    poly_degree: int = 2
    add_polynomial_features: bool = False
    scaler_type: str = "standard"          # "standard" | "robust"
    models: list[str] = field(default_factory=lambda: [
        "linear", "ridge", "lasso", "elasticnet",
        "random_forest", "gradient_boosting"
    ])

# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(
    filepath: Optional[str] = None,
    target_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset from file or fall back to California Housing benchmark.

    Args:
        filepath: Path to a CSV file. If None, uses California Housing dataset.
        target_column: Name of the target column in the CSV.

    Returns:
        X: Feature DataFrame.
        y: Target Series.
    """
    if filepath is not None:
        logger.info("Loading data from %s", filepath)
        df = pd.read_csv(filepath)
        if target_column is None:
            raise ValueError("target_column must be specified when loading from file.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logger.info("Loaded %d rows, %d features", len(df), X.shape[1])
        return X, y

    logger.info("No filepath provided — using California Housing benchmark dataset")
    try:
        housing = fetch_california_housing(as_frame=True)
        X = housing.data
        y = housing.target
        logger.info("Dataset shape: %s | Target: MedHouseVal", X.shape)
    except Exception:
        logger.warning("California Housing download failed — falling back to Diabetes dataset")
        diabetes = load_diabetes(as_frame=True)
        X = diabetes.data
        y = diabetes.target
        logger.info("Dataset shape: %s | Target: disease progression", X.shape)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_features: list[str],
    config: PipelineConfig,
) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric preprocessing.

    Uses RobustScaler when data has significant outliers (interquartile-based),
    StandardScaler otherwise. Polynomial features are optional.

    Args:
        numeric_features: List of numeric column names.
        config: PipelineConfig instance.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    scaler = (
        RobustScaler()
        if config.scaler_type == "robust"
        else StandardScaler()
    )

    if config.add_polynomial_features:
        numeric_transformer = Pipeline(steps=[
            ("scaler", scaler),
            ("poly", PolynomialFeatures(
                degree=config.poly_degree,
                include_bias=False,
                interaction_only=False,
            )),
        ])
        logger.info(
            "Preprocessor: %s + PolynomialFeatures(degree=%d)",
            config.scaler_type,
            config.poly_degree,
        )
    else:
        numeric_transformer = Pipeline(steps=[("scaler", scaler)])
        logger.info("Preprocessor: %s scaler", config.scaler_type)

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop",
    )
    return preprocessor

