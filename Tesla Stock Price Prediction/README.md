<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=20,0,0&height=250&section=header&text=Tesla%20Stock%20AI&fontSize=40&animation=fadeIn&fontAlignY=38&desc=LSTM%20Deep%20Learning%20Forecaster&descAlignY=55&descAlign=50" width="100%" />

  <br/>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
  [![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)](https://github.com/HarshChoudhary2003)

</div>

---

## 🚀 Overview
Predict **Tesla (TSLA)** stock prices with high precision using **LSTM (Long Short-Term Memory)** neural networks. This system combines deep learning with a real-time Streamlit interface for interactive market intelligence.

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
  <h3>⭐ If you found this LSTM forecaster useful, please consider giving it a star!</h3>
  <p>Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a></p>
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=20,0,0&height=100&section=footer" width="100%" />
</div>
