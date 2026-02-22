<div align="center">

# 📊 Stock Price Direction Predictor

### *Support Vector Machines · AI Dashboard · Strategy Backtesting*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/ML%20Type-Classification-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Stock-Reliance%20BSE-orange?style=flat-square" />
<img src="https://img.shields.io/badge/UI-Streamlit%20Dashboard-red?style=flat-square" />

---

*Predict whether Reliance Industries stock will go **UP ↑** or **DOWN ↓** the next day using Support Vector Machines — with live kernel tuning, technical analysis & strategy backtesting.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dashboard Features](#-dashboard-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation & Usage](#-installation--usage)
- [Results](#-results)

---

## 🎯 Overview

This project uses **Support Vector Machines (SVM)** to classify next-day stock price direction (UP or DOWN) for **Reliance Industries (BSE)** based on engineered features. The model is wrapped in a **premium animated Streamlit dashboard** with 5 interactive pages, live hyperparameter tuning, and a full strategy backtest engine.

Key capabilities:
- 🚀 **Live Signal** — Shows BUY/SELL prediction for the next trading day on every page
- 🔬 **Kernel Comparison** — Tests all 4 SVM kernels (Linear, RBF, Poly, Sigmoid) side-by-side
- 💰 **Strategy Backtest** — Compares SVM strategy vs Buy & Hold with cumulative returns and drawdown
- 📡 **Technical Analysis** — RSI, Moving Averages, Volatility, Candlestick charts

---

## 🖥️ Dashboard Features

5 interactive pages in the `app.py` Streamlit dashboard:

| Page | Description |
|------|-------------|
| 🏠 **Overview** | Full price history with MA20/MA50, candlestick (120D), volume bars, return distribution |
| 📈 **Price Analysis** | Multi-panel technical chart (Price + RSI + Volatility), feature space scatter, % UP days gauge |
| 🤖 **SVM Model** | Live model training, confusion matrix, accuracy gauge, Buy/Sell signal overlay, classification report |
| 📊 **Kernel Compare** | Train/Test bars for all 4 kernels, overfitting scatter, leaderboard table, C sweep chart |
| 💰 **Backtest** | SVM vs Buy & Hold cumulative returns, alpha, win rate, daily returns bar, drawdown chart |

### 🎛️ Sidebar Controls
- **SVM Kernel** — rbf / linear / poly / sigmoid
- **C** (Regularization strength)
- **Gamma** — scale / auto / manual values
- **Train Split %** — 60–90%
- **Feature Scaling** toggle (StandardScaler on/off)

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `RELIANCE.csv` |
| **Exchange** | BSE (Bombay Stock Exchange) |
| **Rows** | ~2,634 trading days |
| **Columns** | Date, Open, High, Low, Close, Adj Close, Volume |

---

## 🔬 Methodology

```
┌─────────────┐    ┌─────────────────────┐    ┌─────────────┐
│   RELIANCE  │ -> │  Feature Engineering │ -> │  SVM        │
│   .csv      │    │  Open-Close          │    │ Classifier  │
│             │    │  High-Low            │    │ (Train/Test)│
└─────────────┘    └─────────────────────┘    └─────────────┘
                                                      │
┌─────────────┐    ┌─────────────┐    ┌──────────────▼──────┐
│  Strategy   │<-  │  Signals    │<-  │  Predicted           │
│  Backtest   │    │  BUY / SELL │    │  Direction (0 or 1)  │
└─────────────┘    └─────────────┘    └─────────────────────┘
```

### Feature Engineering
- **Open-Close** = Open price − Close price
- **High-Low** = High price − Low price
- **Target** = 1 (UP) if next-day Close > today's Close, else 0 (DOWN)

### Technical Indicators (Dashboard)
- 📈 MA20 / MA50 moving averages
- 📉 RSI (14-period Relative Strength Index)
- 📊 20-day rolling Volatility

### SVM Kernels Tested
| Kernel | Description |
|--------|-------------|
| **RBF** | Radial Basis Function — default, handles non-linear boundaries |
| **Linear** | Best for linearly separable data |
| **Polynomial** | Captures polynomial feature interactions (degree=3) |
| **Sigmoid** | Similar to neural network activation |

---

## 🚀 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Predicting Stock Price Direction using Support Vector Machines"

# Install dependencies
pip install -r requirements.txt
```

### ▶️ Run Streamlit Dashboard
```bash
streamlit run app.py
```

### 📓 Run Jupyter Notebook
```bash
jupyter notebook main.ipynb
```

---

## 📈 Results

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of days correctly classified as UP or DOWN |
| **Precision** | Of predicted UPs, how many were actually UP |
| **Recall** | Of actual UPs, how many were correctly predicted |
| **F1-Score** | Harmonic mean of Precision and Recall |

*Live metrics with confusion matrix and classification report available in the 🤖 Model page.*

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
