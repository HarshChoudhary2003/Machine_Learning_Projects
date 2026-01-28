<div align="center">

# 🪙 Bitcoin Price Prediction

### *Machine Learning for Cryptocurrency Forecasting*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Crypto-Bitcoin-F7931A?style=flat-square" />
<img src="https://img.shields.io/badge/ML%20Type-Time%20Series-blue?style=flat-square" />

---

*Predict Bitcoin price movements using machine learning on historical cryptocurrency data.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Installation](#-installation)
- [Methodology](#-methodology)
- [Results](#-results)

---

## 🎯 Overview

Bitcoin and cryptocurrency markets are highly volatile, making them challenging yet interesting for price prediction. This project applies **machine learning techniques** to forecast Bitcoin price movements.

### 🌟 Key Objectives
- 📈 Predict Bitcoin price trends
- 📊 Analyze historical patterns
- 🔍 Feature engineering for crypto
- 🧪 Model experimentation

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `bitcoin.csv` |
| **Size** | ~221 KB |
| **Features** | Open, High, Low, Close, Volume |
| **Timeframe** | Historical BTC/USD data |

---

## ✨ Features Analyzed

| Feature | Type |
|---------|------|
| 📈 **Open** | Daily opening price |
| 📊 **High** | Daily high |
| 📉 **Low** | Daily low |
| 💰 **Close** | Daily closing price |
| 📦 **Volume** | Trading volume |

### Engineered Features
- 📅 Temporal features (day, month, year)
- 🔄 Lag features
- 📊 Moving averages (SMA, EMA)
- 📈 Price momentum
- 💹 Volatility indicators

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Bitcoin Price Prediction using Machine Learning in Python"

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

---

## 💻 Usage

```bash
# Launch Jupyter Notebook
jupyter notebook main.ipynb
```

---

## 🔬 Methodology

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │ -> │   Feature   │ -> │   Model     │
│   Loading   │    │   Eng.      │    │   Training  │
└─────────────┘    └─────────────┘    └─────────────┘
                                             │
         ┌───────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────┐
│              PREDICTION & EVALUATION            │
└─────────────────────────────────────────────────┘
```

---

## 📈 Results

| Metric | Description |
|--------|-------------|
| **R² Score** | Model fit quality |
| **RMSE** | Prediction error |
| **MAE** | Average error |

*Detailed results in notebook*

---

## ⚠️ Disclaimer

> **For educational purposes only.** Cryptocurrency markets are extremely volatile. This project is not financial advice. Always do your own research.

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
