<div align="center">

# 💳 EMI Calculator & Prediction

### *Machine Learning for Loan EMI Analysis*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-Finance-blue?style=flat-square" />
<img src="https://img.shields.io/badge/MLOps-MLflow-purple?style=flat-square" />

---

*A comprehensive EMI prediction system with MLflow experiment tracking and web interface.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [MLOps Integration](#-mlops-integration)

---

## 🎯 Overview

This project builds an **EMI (Equated Monthly Installment)** prediction system using machine learning. It includes proper MLOps practices with **MLflow** for experiment tracking and model versioning.

### 🌟 Key Features
- 💰 **EMI Prediction** - Accurate installment estimation
- 📊 **MLflow Tracking** - Experiment management
- 🌐 **Web Interface** - User-friendly app
- 📈 **Data Analysis** - Comprehensive EDA

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💵 **EMI Calculation** | Predict monthly installments |
| 📊 **Model Training** | Multiple ML algorithms |
| 🔄 **MLflow** | Track experiments & models |
| 🌐 **Streamlit App** | Interactive web interface |
| 📈 **Visualization** | Financial insights |

---

## 📁 Project Structure

```
EMI/
├── app/                    # Streamlit application
│   └── app.py
├── data/                   # Dataset files
│   ├── raw/
│   └── processed/
├── notebooks/              # Jupyter notebooks
│   └── eda.ipynb
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── prediction.py
├── mlflow/                 # MLflow configuration
├── mlruns/                 # MLflow runs
├── requirements.txt        # Dependencies
├── runtime.txt             # Python version
└── README.md
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/EMI"

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Run Notebooks
```bash
jupyter notebook notebooks/eda.ipynb
```

### Launch Web App
```bash
streamlit run app/app.py
```

### View MLflow Dashboard
```bash
mlflow ui
# Open http://localhost:5000
```

---

## 📊 MLOps Integration

### MLflow Features Used
- 📝 **Experiment Tracking** - Log parameters & metrics
- 💾 **Model Registry** - Version control for models
- 📈 **Metric Comparison** - Compare model performance
- 🔄 **Reproducibility** - Track all experiments

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("rmse", rmse_score)
    mlflow.sklearn.log_model(model, "model")
```

---

## 🔢 EMI Formula

```
EMI = [P × R × (1+R)^N] / [(1+R)^N - 1]

Where:
P = Principal loan amount
R = Monthly interest rate
N = Number of monthly installments
```

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
