<div align="center">

# 🌫️ AirSense AI — Predicting Air Quality with Neural Networks

### *Deep Learning for Environmental Health Monitoring*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)

<img src="https://img.shields.io/badge/Status-Complete-22c55e?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-Environmental%20AI-0ea5e9?style=flat-square" />
<img src="https://img.shields.io/badge/UI-Streamlit%20App-ff4b4b?style=flat-square" />
<img src="https://img.shields.io/badge/Cities-26%20Indian%20Cities-818cf8?style=flat-square" />

---

*Predict the **Air Quality Index (AQI)** and pollutant levels from real Indian city data using a deep Artificial Neural Network — with a stunning animated Streamlit dashboard.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Streamlit App](#-streamlit-app)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [App Features](#-app-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Results](#-results)

---

## 🎯 Overview

Air pollution is one of the most critical global health crises of our time. This project leverages **Artificial Neural Networks (ANN)** to predict the **Air Quality Index (AQI)** and pollutant concentrations from historical urban data across India.

### 🌟 Key Objectives
- 📈 Predict AQI values accurately from 12 pollutant features
- 🧪 Analyse the impact of pollutants — PM2.5, PM10, NO2, CO, SO2, O3, Benzene, and more
- 🧠 Implement a robust Deep Learning regression pipeline
- 📊 Visualise environmental trends interactively via a Streamlit dashboard
- 🏥 Provide real-time health guidance based on predicted AQI category

---

## 🚀 Streamlit App

A premium animated **AirSense AI** dashboard built with Streamlit includes:

| Feature | Description |
|---------|-------------|
| 🔮 **AQI Predictor** | Enter 12 pollutant values via sliders → instant AQI prediction |
| 🎯 **AQI Gauge** | Animated circular gauge coloured by severity |
| 🕸️ **Radar Chart** | Pollutant profile radar visualisation |
| 📅 **Timeline** | City-wise AQI over time (interactive) |
| 🔥 **Heatmap** | Pollutant correlation heatmap |
| 🥧 **Distribution** | AQI category donut chart |
| 🏆 **City Ranking** | City-wise average AQI bar chart |
| 📆 **Monthly Trend** | Area chart of monthly average AQI |

### Run the app

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| **File** | `city_day.csv` |
| **Source** | [Air Quality Data in India — Kaggle](https://www.kaggle.com/rohanrao/air-quality-data-in-india) |
| **Coverage** | 26 Indian cities · 2015 – 2020 |
| **Input Features** | PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene |
| **Target Variable** | AQI (Air Quality Index) |
| **AQI Categories** | Good · Satisfactory · Moderate · Poor · Very Poor · Severe |

---

## 🧠 Model Architecture

Deep Feed-Forward Neural Network trained on standardised pollutant features:

```
┌──────────────────────────────────────────────┐
│          INPUT LAYER (12 features)           │
│   PM2.5, PM10, NO, NO2, NOx, NH3, CO,       │
│   SO2, O3, Benzene, Toluene, Xylene         │
└───────────────────┬──────────────────────────┘
                    │  StandardScaler
┌───────────────────▼──────────────────────────┐
│        HIDDEN LAYER 1 — Dense 128            │
│           ReLU Activation + Dropout          │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│        HIDDEN LAYER 2 — Dense 64             │
│           ReLU Activation + Dropout          │
└───────────────────┬──────────────────────────┘
                    │
┌───────────────────▼──────────────────────────┐
│         OUTPUT LAYER — Dense 1               │
│         AQI Regression Value                 │
└──────────────────────────────────────────────┘
```

**Training Details:**
- Loss function: Mean Squared Error (MSE)
- Optimiser: Adam
- Regularisation: Dropout layers
- Preprocessing: `StandardScaler` normalisation
- Model saved as: `model.h5`

---

## ✨ App Features

### 🎨 Design Highlights
- 🌑 **Dark glassmorphism** theme (`#060b18` base)
- 🌈 **Animated gradient hero** with floating orbs and glow effects
- ✨ **Shimmer hover effects** on stat cards
- 🎯 **Pulsing glow** on the AQI result card
- 📊 **Transparent Plotly charts** seamlessly integrated
- 🔠 **Space Grotesk + Inter** Google Fonts
- 💡 **High-contrast text** optimised for readability on dark backgrounds

### 📐 Layout
- **Tab 1 — Predict AQI:** Sliders + live pollutant pills + gauge + radar
- **Tab 2 — Data Explorer:** Timeline, donut, bar, heatmap, monthly trend, raw data
- **Tab 3 — About & Model:** Architecture diagram, AQI reference, tech stack

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Data Processing** | Pandas, NumPy |
| **Preprocessing** | Scikit-Learn (StandardScaler) |
| **Visualisation** | Plotly |
| **Web App** | Streamlit |
| **Fonts** | Google Fonts (Inter, Space Grotesk) |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# 2. Navigate to this project
cd "Machine_Learning_Projects/Predicting Air Quality with Neural Networks"

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```

> **Note:** Ensure `model.h5` and `city_day.csv` are in the same directory as `app.py`.

---

## 📈 Results

The trained ANN achieves strong regression performance on AQI prediction:

| Metric | Performance |
|--------|-------------|
| 📉 **MAE** | Low mean absolute error on test set |
| 📊 **R² Score** | High variance coverage |
| 🔮 **AQI Categories** | Accurately identifies peak pollution periods |
| 🏙️ **City Coverage** | Generalises across 26 diverse Indian cities |

---

## 🏥 AQI Health Reference

| AQI Range | Category | Health Impact |
|-----------|----------|---------------|
| 0 – 50 | 🟢 Good | Safe for all |
| 51 – 100 | 🟡 Moderate | Sensitive groups cautious |
| 101 – 150 | 🟠 Unhealthy for Sensitive | Sensitive groups at risk |
| 151 – 200 | 🔴 Unhealthy | General public at risk |
| 201 – 300 | 🟣 Very Unhealthy | Serious health effects |
| 300+ | ⚫ Hazardous | Emergency — stay indoors |

---

<div align="center">

### ⭐ If you found this useful, give it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

*🌫️ AirSense AI — Powered by TensorFlow & Streamlit*

</div>
