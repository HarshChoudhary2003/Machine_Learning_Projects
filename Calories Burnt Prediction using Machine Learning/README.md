# 🔥 Calories Burnt Prediction using Machine Learning

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit%20Learn-orange?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/XGBoost-Regressor-green?style=for-the-badge&logo=xgboost" />
</div>

## 📌 Overview
This project aims to build a machine learning model to predict the amount of **calories burnt** during exercise based on various physiological and activity-based factors. This kind of prediction is crucial for personal fitness tracking apps and health monitoring systems.

## 📂 Dataset Details
The dataset contains information about individuals' body measurements and their exercise intensity.

- **Gender**: Male/Female
- **Age**: Age of the person (years)
- **Height**: Height (cm)
- **Weight**: Weight (kg)
- **Duration**: Duration of exercise (minutes)
- **Heart_Rate**: Heart rate during exercise (bpm)
- **Body_Temp**: Body temperature (°C)
- **Calories**: Calories burnt (Target Variable)

## 🛠️ Technologies Used
- **Python** 🐍
- **Pandas** (Data Manipulation)
- **NumPy** (Numerical Computations)
- **Matplotlib & Seaborn** (Data Visualization)
- **Scikit-Learn** (Model Building & Evaluation)
- **XGBoost** (Advanced Gradient Boosting)

## 🤖 Models Exploring
The project explores multiple regression algorithms to find the best predictor:
1. **Lasso Regression**
2. **Ridge Regression**
3. **Linear Regression**
4. **Support Vector Regressor (SVR)**
5. **Random Forest Regressor**
6. **XGBoost Regressor** (Likely the best performer)

## 🚀 Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   ```
2. Navigate to the project directory:
   ```bash
   cd "Calories Burnt Prediction using Machine Learning"
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

## 📊 Result
The models are evaluated based on metrics like **Mean Absolute Error (MAE)** and **R² Score** to ensure accurate predictions.

---
<div align="center">
Made with ❤️ by Harsh Choudhary
</div>
