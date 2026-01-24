"""
========================================================
EMIPredict AI - Feature Preprocessing Pipeline
========================================================
Author: Harsh Chouhary
Description:
  Builds ColumnTransformer for numeric scaling and categorical encoding.
  Used in both classification and regression training scripts.
========================================================
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def build_preprocessor(df):
    """
    Automatically detects numeric and categorical features.
    Returns a scikit-learn ColumnTransformer.
    """
    numeric_features = [
        "age","monthly_salary","years_of_employment","monthly_rent","family_size",
        "dependents","total_monthly_expenses","dti_ratio","affordability_ratio",
        "log_bank_balance","log_emergency_fund","current_emi_amount",
        "credit_score","bank_balance","emergency_fund","requested_amount","requested_tenure"
    ]
    numeric_features = [col for col in numeric_features if col in df.columns]

    categorical_features = [
        "gender","marital_status","education","employment_type","company_type",
        "house_type","emi_scenario"
    ]
    categorical_features = [col for col in categorical_features if col in df.columns]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor, numeric_features, categorical_features
