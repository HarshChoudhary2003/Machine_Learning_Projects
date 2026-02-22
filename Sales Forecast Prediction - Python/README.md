<div align="center">

# 📈 Sales Forecast Prediction

### *Interactive AI Dashboard · Time Series Forecasting with XGBoost*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/ML%20Type-Time%20Series-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Industry-Retail-green?style=flat-square" />
<img src="https://img.shields.io/badge/UI-Streamlit%20Dashboard-red?style=flat-square" />

---

*Predict future sales trends to optimize inventory and maximize revenue using machine learning — now with a full interactive Streamlit dashboard.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dashboard Features](#-dashboard-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Installation & Usage](#-installation--usage)
- [Results](#-results)
- [Business Impact](#-business-impact)

---

## 🎯 Overview

Accurate sales forecasting is critical for business success. This project develops **machine learning models** to predict future sales based on historical retail data, wrapped in a **premium animated Streamlit dashboard** with 5 dedicated pages.

It helps businesses:

- 📦 **Optimize Inventory** — Reduce overstock and stockouts
- 💰 **Improve Cash Flow** — Better financial planning
- 📊 **Strategic Planning** — Data-driven decisions
- 🎯 **Target Setting** — Realistic sales goals
- 🔮 **Future Forecasting** — Up to 90-day rolling predictions

---

## 🖥️ Dashboard Features

The interactive `app.py` Streamlit dashboard includes **5 pages**:

| Page | Description |
|------|-------------|
| 🏠 **Overview** | KPI cards, Sales timeline with moving averages, Monthly heatmap, Category pie |
| 📊 **EDA** | Region/sub-category bars, Sales distribution, Day-of-week patterns, Quarterly trends |
| 🤖 **Model** | Live XGBoost training, Actual vs Predicted, Residual analysis, Feature importance |
| 🔮 **Forecast** | Configurable 7–90 day future forecast with confidence bands + CSV download |
| 📦 **Insights** | Customer segments, Regional sunburst, Seasonal polar chart, YoY growth |

### ⚙️ Interactive Controls (Sidebar)
- **Filter** by Year, Category, Region
- **Tune** lag features, n_estimators, learning rate, max depth
- **Set** forecast horizon (7–90 days)

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `train.csv` |
| **Size** | ~2.1 MB · 9,800 rows |
| **Key Features** | Order Date, Sales, Category, Region, Segment, Ship Mode |
| **Date Format** | DD/MM/YYYY |

---

## 🔬 Methodology

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Data      │ -> │   Feature   │ -> │  Lag / MA   │
│   Loading   │    │   Eng.      │    │  Features   │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
┌─────────────┐    ┌─────────────┐    ┌──────▼──────┐
│  Forecast   │ <- │   XGBoost   │ <- │  Train /    │
│  Dashboard  │    │   Training  │    │  Test Split │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Feature Engineering
- 📅 Day, Month, Year, Quarter, Week-of-Year extraction
- 📆 Day-of-week patterns
- 🔄 Configurable lag features (previous N days' sales)
- 📈 7-day & 30-day rolling averages

### Model
- 🚀 **XGBoost Regressor** — `reg:squarederror` objective
- Tunable: `n_estimators`, `learning_rate`, `max_depth`, `subsample`
- Metrics: **R²**, **RMSE**, **MAE**, **MAPE**

---

## 🚀 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Sales Forecast Prediction - Python"

# Install dependencies
pip install -r requirements.txt
```

### ▶️ Run Streamlit Dashboard
```bash
streamlit run app.py
```

### 📓 Run Jupyter Notebook (EDA)
```bash
jupyter notebook main.ipynb
```

---

## 📈 Results

| Metric | Description |
|--------|-------------|
| **R²** | Coefficient of determination — goodness of fit |
| **RMSE** | Root Mean Square Error |
| **MAE** | Mean Absolute Error |
| **MAPE** | Mean Absolute Percentage Error |

*Live metrics available interactively in the dashboard's 🤖 Model page.*

---

## 💼 Business Impact

| Benefit | Impact |
|---------|--------|
| 📉 **Reduced Waste** | Lower inventory costs |
| 📈 **Increased Sales** | Better stock availability |
| 💵 **Cost Savings** | Optimized supply chain |
| 📊 **Better Planning** | Accurate budgeting |
| 🔮 **Future-Ready** | 90-day rolling forecast |

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
