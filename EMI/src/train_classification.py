import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from features import build_preprocessor
from mlflow.models import infer_signature

EXPERIMENT_NAME = "EMIPredict_Classification"


# ----------------------------- Utility -----------------------------

def load_split(split_path):
    """Load parquet dataset"""
    return pd.read_parquet(split_path)


def evaluate(y_true, y_pred, y_prob=None):
    """Compute classification metrics"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 2:
        try:
            metrics["roc_auc"] = roc_auc_score(pd.get_dummies(y_true), y_prob, multi_class="ovr")
        except Exception:
            pass
    return metrics


def train_and_log(model_name, model, X_train, y_train, X_val, y_val, preprocessor, label_mapping):
    """Train model, log to MLflow, and return metrics"""
    with mlflow.start_run(run_name=model_name):
        pipe = Pipeline(steps=[("preproc", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        probs = (
            pipe.predict_proba(X_val)
            if hasattr(pipe.named_steps["model"], "predict_proba")
            else None
        )

        metrics = evaluate(y_val, preds, probs)
        mlflow.log_params({"model": model_name})

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Log label mapping
        mlflow.log_dict(label_mapping, "label_mapping.json")

        # Log model with signature for clean UI
        try:
            signature = infer_signature(X_val, preds)
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


# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load datasets
    train = load_split("data/processed/train.parquet")
    val = load_split("data/processed/val.parquet")

    # Separate features and targets
    X_train = train.drop(columns=["emi_eligibility", "max_monthly_emi"])
    y_train_raw = train["emi_eligibility"]
    X_val = val.drop(columns=["emi_eligibility", "max_monthly_emi"])
    y_val_raw = val["emi_eligibility"]

    # Label encoding for XGBoost compatibility
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_val = label_encoder.transform(y_val_raw)

    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    preprocessor, _, _ = build_preprocessor(train)

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            num_class=len(np.unique(y_train))
        ),
    }

    # Train and log all models
    for name, model in models.items():
        train_and_log(name, model, X_train, y_train, X_val, y_val, preprocessor, label_mapping)
