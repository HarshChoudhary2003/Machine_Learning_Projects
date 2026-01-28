<div align="center">

# 📈 Microsoft Stock Price Prediction

### *Deep Learning for Financial Forecasting*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/ML%20Type-Time%20Series-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Stock-MSFT-0078D4?style=flat-square" />

---

*Predict Microsoft (MSFT) stock prices using LSTM neural networks and historical market data.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Results](#-results)

---

## 🎯 Overview

This project applies **Long Short-Term Memory (LSTM)** networks to predict Microsoft stock prices. LSTM networks are particularly well-suited for time series prediction due to their ability to learn long-term dependencies.

### 🌟 Key Features
- 📊 Historical stock data analysis
- 🧠 Deep learning with LSTM
- 📈 Trend visualization
- 🔮 Future price prediction

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `MicrosoftStock.csv` |
| **Size** | ~70 KB |
| **Features** | Open, High, Low, Close, Volume |
| **Stock** | Microsoft (MSFT) |

---

## 🧠 Model Architecture

```
┌─────────────────────────────────────────┐
│           INPUT LAYER                   │
│         (Sequence Data)                 │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           LSTM LAYER 1                  │
│         (64/128 units)                  │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           LSTM LAYER 2                  │
│         (32/64 units)                   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           DENSE LAYER                   │
│         (Output: 1)                     │
└─────────────────────────────────────────┘
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Microsoft Stock Price Prediction with Machine Learning"

# Install dependencies
pip install pandas numpy tensorflow keras matplotlib scikit-learn jupyter
```

---

## 💻 Usage

```bash
# Launch Jupyter Notebook
jupyter notebook main.ipynb
```

---

## 📈 Results

### Prediction Visualization
The model provides visual comparison between:
- 📊 Actual stock prices
- 🔮 Predicted stock prices

### Metrics
| Metric | Description |
|--------|-------------|
| **RMSE** | Prediction accuracy |
| **Loss** | Training/validation loss |

---

## ⚠️ Disclaimer

> This project is for **educational purposes only**. Stock market predictions are inherently uncertain. Do not use this for actual trading decisions.

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
