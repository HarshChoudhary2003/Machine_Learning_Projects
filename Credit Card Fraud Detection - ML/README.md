<div align="center">

# 💳 Credit Card Fraud Detection

### *Machine Learning for Financial Security*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/ML%20Type-Classification-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-Finance-green?style=flat-square" />

---

*Detect fraudulent credit card transactions using machine learning with imbalanced data handling.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Challenge](#-challenge)
- [Dataset](#-dataset)
- [Techniques](#-techniques)
- [Installation](#-installation)
- [Results](#-results)

---

## 🎯 Overview

Credit card fraud costs billions annually. This project builds a **machine learning classifier** to detect fraudulent transactions in real-time, helping financial institutions prevent fraud and protect customers.

### 🌟 Key Objectives
- 🔍 Detect fraudulent transactions
- ⚖️ Handle severe class imbalance
- ⚡ Real-time classification capability
- 📊 Minimize false positives/negatives

---

## ⚠️ Challenge

### The Imbalance Problem

```
┌────────────────────────────────────────────┐
│          TRANSACTION DISTRIBUTION          │
├────────────────────────────────────────────┤
│                                            │
│  Legitimate: ███████████████████████ 99.8% │
│  Fraudulent: ░                       0.2%  │
│                                            │
└────────────────────────────────────────────┘
```

Fraud detection is challenging because:
- 📊 **Imbalanced Data** - Very few fraud cases
- 🎭 **Evolving Patterns** - Fraudsters adapt
- ⏱️ **Real-time Need** - Instant decisions required
- 💰 **High Stakes** - Wrong predictions are costly

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `creditcard.csv` |
| **Size** | ~150 MB |
| **Records** | 284,807 transactions |
| **Frauds** | 492 (0.17%) |
| **Features** | PCA-transformed (V1-V28) |

---

## 🔬 Techniques

### Handling Imbalance
| Technique | Description |
|-----------|-------------|
| 📈 **SMOTE** | Synthetic oversampling |
| 📉 **Undersampling** | Reduce majority class |
| ⚖️ **Class Weights** | Penalize misclassification |

### Models Used
- 🌲 Random Forest
- 📊 Logistic Regression
- 🚀 XGBoost
- 🎯 Isolation Forest

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Credit Card Fraud Detection - ML"

# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost jupyter
```

---

## 💻 Usage

```bash
# Launch Jupyter Notebook
jupyter notebook main.ipynb
```

---

## 📈 Results

### Performance Metrics

| Metric | Importance |
|--------|------------|
| **Recall** | Catch all frauds (most important) |
| **Precision** | Avoid false alarms |
| **F1-Score** | Balance of both |
| **AUC-ROC** | Overall discrimination |

### Confusion Matrix Analysis
```
                    Predicted
                 Fraud    Legit
Actual  Fraud     TP        FN  ← Minimize (missed fraud)
        Legit     FP        TN  ← Minimize (false alarm)
```

---

## 💼 Business Impact

| Outcome | Value |
|---------|-------|
| 💰 **Fraud Prevented** | Millions saved |
| 😊 **Customer Trust** | Enhanced security |
| ⚡ **Real-time** | Instant protection |

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
