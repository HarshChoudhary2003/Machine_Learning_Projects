import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature
from features import build_preprocessor

EXPERIMENT_NAME = "EMIPredict_Regression"

# ----------------------------- Utility ----------------------------- #

def load_split(split_path):
    """Load parquet dataset safely"""
    return pd.read_parquet(split_path)

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1, y_true)  # avoid division by zero
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate(y_true, y_pred):
    """Compute regression metrics compatible with all sklearn versions"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mape(y_true, y_pred))
    }

def train_and_log(model_name, model, X_train, y_train, X_val, y_val, preprocessor):
    """Train, evaluate, and log model with MLflow"""
    with mlflow.start_run(run_name=model_name):
        # Build full pipeline
        pipe = Pipeline(steps=[("preproc", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)

        # Evaluate model
        metrics = evaluate(y_val, preds)

        # Log model parameters & metrics
        mlflow.log_params({"model": model_name})
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log model cleanly with schema
        try:
            signature = infer_signature(X_val.head(5), preds[:5])
            mlflow.sklearn.log_model(
                pipe,
                name=f"{model_name}_model",
                input_example=X_val.head(1),
                signature=signature,
            )
        except Exception:
            mlflow.sklearn.log_model(pipe, name=f"{model_name}_model")

        print(f"\n {model_name}: {metrics}\n")
        return metrics


# ----------------------------- Main ----------------------------- #

if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load processed splits
    train = load_split("data/processed/train.parquet")
    val = load_split("data/processed/val.parquet")

    # Optionally sample for speed (remove this once testing is done)
    # train = train.sample(frac=0.15, random_state=42)
    # val = val.sample(frac=0.15, random_state=42)

    # Split features & target
    X_train = train.drop(columns=["emi_eligibility", "max_monthly_emi"])
    y_train = train["max_monthly_emi"]
    X_val = val.drop(columns=["emi_eligibility", "max_monthly_emi"])
    y_val = val["max_monthly_emi"]

    # Build preprocessing pipeline
    preprocessor, _, _ = build_preprocessor(train)

    # Define regression models (optimized for performance)
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=80,         # Reduced from 200 for speed
            max_depth=20,            # Avoid deep trees
            min_samples_split=4,
            n_jobs=-1,               # Use all CPU cores
            random_state=42
        ),
        "XGBoost": XGBRegressor(
            random_state=42,
            n_estimators=200,        # Balanced speed & performance
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=-1
        ),
    }

    # Train and log each model
    for name, model in models.items():
        train_and_log(name, model, X_train, y_train, X_val, y_val, preprocessor)
