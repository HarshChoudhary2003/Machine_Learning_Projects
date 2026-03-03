<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=20,0,0&height=250&section=header&text=Credit%20Card%20Fraud&fontSize=40&animation=fadeIn&fontAlignY=38&desc=Machine%20Learning%20for%20Financial%20Security&descAlignY=55&descAlign=50" width="100%" />

  <br/>

  [![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

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
  <h3>⭐ If you found this project useful, please consider giving it a star!</h3>
  <p>Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a></p>
  
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=20,0,0&height=100&section=footer" width="100%" />
</div>
