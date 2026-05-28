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
