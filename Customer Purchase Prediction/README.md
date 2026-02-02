# Customer Purchase Prediction 🛍️

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-red.svg)](https://pandas.pydata.org/)

Predict whether a customer will make a purchase based on their demographic and behavioral data using Machine Learning classification algorithms.

## 📌 Project Overview

In the competitive world of e-commerce, understanding customer behavior is key to driving sales. This project implements a classification model to predict the `purchased` status of a customer. By analyzing features such as age, income, and time spent on the site, businesses can target potential buyers more effectively with personalized marketing.

## 📊 Dataset Description

The dataset `customer_purchase_data.csv` contains information about 1,000 customers.

**Features:**
- `age`: Age of the customer.
- `income`: Annual income of the customer.
- `gender`: Gender (Male/Female).
- `city`: City of residence (Bangalore, Chennai, Delhi, Mumbai).
- `time_on_site`: Average time spent on the website (minutes).
- `pages_viewed`: Number of pages viewed during the session.
- `purchased`: Target variable (1 = Yes, 0 = No).

## 🧠 Methodology

1.  **Exploratory Data Analysis (EDA):** Visualizing distributions and correlations between demographic factors and purchase behavior.
2.  **Data Preprocessing:**
    - handling categorical variables using One-Hot Encoding.
    - Feature scaling for numerical attributes.
3.  **Model Selection:**
    - **Logistic Regression:** Used as a baseline model for binary classification.
    - **Random Forest Classifier:** Implemented for better handling of non-linear relationships.
4.  **Optimization:**
    - Hyperparameter tuning to improve generalization.
    - Probability threshold tuning to balance **Precision** and **Recall** based on business needs.

## 🛠️ Technology Stack

- **Language:** `Python`
- **Libraries:** `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-Learn`

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Customer Purchase Prediction"
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Explore the Analysis:**
   Open the relevant Jupyter notebook or Python script to view the model training process and evaluation metrics.

## 📈 Key Results

- Identified `income` and `pages_viewed` as top predictors for purchase behavior.
- Random Forest achieved significantly higher F1-score compared to the baseline Logistic Regression model.

---
Developed with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)
