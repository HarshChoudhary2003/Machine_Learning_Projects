#!/bin/bash
# =======================================================
# EMIPredict AI - MLflow Tracking Server
# =======================================================
# Author: Harsh Chouhary
# Description:
#   Launches MLflow tracking server with SQLite backend.
#   Stores all experiment runs and model artifacts.
# =======================================================

echo " Starting MLflow Tracking Server..."

# Set artifact and backend paths
export MLFLOW_BACKEND_URI="sqlite:///mlflow/mlflow.db"
export MLFLOW_ARTIFACT_ROOT="./mlruns"

# Run the MLflow server
mlflow server \
  --backend-store-uri $MLFLOW_BACKEND_URI \
  --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
  --host 0.0.0.0 \
  --port 5000
