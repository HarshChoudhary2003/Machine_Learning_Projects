<div align="center">

# 🚀 Tesla Stock Price Prediction

### *LSTM Deep Learning for TSLA Forecasting*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Stock-TSLA-CC0000?style=flat-square" />
<img src="https://img.shields.io/badge/Deep%20Learning-LSTM-purple?style=flat-square" />

---

*Predict Tesla (TSLA) stock prices using deep learning with an interactive Streamlit web application.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)

---

## 🎯 Overview

This project predicts Tesla stock prices using **LSTM (Long Short-Term Memory)** neural networks. It includes both a Jupyter notebook for model development and a **Streamlit web app** for interactive predictions.

### 🌟 Key Highlights
- 🧠 **Deep Learning** - LSTM for time series
- 🌐 **Web Interface** - Interactive Streamlit app
- 📊 **Visualization** - Real-time price charts
- 💾 **Pre-trained Models** - Ready to use

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📈 **Price Prediction** | Forecast TSLA prices |
| 🌐 **Web App** | Interactive dashboard |
| 📊 **Charts** | Historical vs predicted |
| 🔮 **Future Forecasting** | Multi-day predictions |

---

## 📁 Project Structure

```
Tesla Stock Price Prediction/
├── tesla_stock_prediction.ipynb  # Model development notebook
├── app.py                         # Streamlit web application
├── TSLA.csv                       # Historical stock data
├── models/                        # Saved model files
│   ├── model.h5                   # Trained LSTM model
│   └── scaler.pkl                 # Data scaler
└── README.md                      # This file
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Tesla Stock Price Prediction"

# Install dependencies
pip install pandas numpy tensorflow keras streamlit matplotlib scikit-learn jupyter
```

---

## 💻 Usage

### Run Jupyter Notebook
```bash
jupyter notebook "tesla_stock_prediction.ipynb"
```

### Launch Web App
```bash
streamlit run app.py
```

---

## 🧠 Model Details

### Architecture
```
Input (60 time steps) → LSTM(128) → Dropout(0.2) → LSTM(64) → Dense(1)
```

### Training Parameters
| Parameter | Value |
|-----------|-------|
| **Epochs** | 100 |
| **Batch Size** | 32 |
| **Optimizer** | Adam |
| **Loss** | MSE |

---

## 📈 Results

- 📊 Visual comparison of actual vs predicted prices
- 📉 Trend analysis and accuracy metrics
- 🔮 Future price projections

---

## ⚠️ Disclaimer

> **Educational purposes only.** This is not financial advice. Stock markets are volatile and unpredictable. Always consult professional advisors.

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
