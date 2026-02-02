<div align="center">

# 🌫️ Predicting Air Quality with Neural Networks

### *Deep Learning for Environmental Monitoring*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-Environmental-green?style=flat-square" />
<img src="https://img.shields.io/badge/Deep%20Learning-Neural%20Networks-blue?style=flat-square" />

---

*Predict air quality levels and pollutant concentrations using deep neural networks to help monitor environmental health.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Results](#-results)

---

## 🎯 Overview

Air pollution is a global health crisis. This project leverages **Artificial Neural Networks (ANN)** to predict the **Air Quality Index (AQI)** and pollutant levels based on historical urban data. By accurately forecasting air quality, this system can provide early warnings for health-sensitive populations.

### 🌟 Key Objectives
- 📈 Predict AQI values with high accuracy
- 🧪 Analyze the impact of various pollutants (PM2.5, NO2, CO, etc.)
- 🧠 Implement robust Deep Learning architectures
- 📊 Visualize environmental trends over time

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `city_day.csv` |
| **Source** | [Air Quality Data in India](https://www.kaggle.com/rohanrao/air-quality-data-in-india) |
| **Purity Metrics** | PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene |
| **Target Variable** | AQI (Air Quality Index) |

---

## 🧠 Model Architecture

The core of this project is a Deep Feed-Forward Neural Network:

```
┌─────────────────────────────────────────┐
│           INPUT LAYER                   │
│         (Pollutant Features)            │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           HIDDEN LAYER 1                │
│       (Dense + ReLU + Dropout)          │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           HIDDEN LAYER 2                │
│       (Dense + ReLU + Dropout)          │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│           OUTPUT LAYER                  │
│        (AQI Regression Value)           │
└─────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | technologies |
|----------|--------------|
| **Deep Learning** | TensorFlow, Keras |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn |

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Predicting Air Quality with Neural Networks"

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

---

## 📈 Results

The model achieves strong performance in regression tasks for AQI prediction:
- 📉 **Low MAE/MSE** on test datasets
- 📊 **High R² Score** indicating good variance coverage
- 🔮 Capable of identifying peak pollution periods

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
