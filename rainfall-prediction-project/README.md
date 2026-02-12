# 🌧️ Rainfall Prediction Project

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Matplotlib">
</div>

## 📌 Overview
This project focuses on predicting rainfall using **Linear Regression** on the Austin weather dataset. By analyzing various atmospheric parameters such as temperature, humidity, dew point, and wind speed, the model aims to estimate precipitation levels.

## 📂 Dataset
The dataset used is `austin_weather.csv`, which contains historical weather data for Austin, Texas.
- **Key Features:**
  - `TempHighF`, `TempAvgF`, `TempLowF`: Temperature values (High, Avg, Low)
  - `DewPointHighF`, `DewPointAvgF`, `DewPointLowF`: Dew point values
  - `HumidityHighPercent`, `HumidityAvgPercent`, `HumidityLowPercent`: Humidity percentages
  - `SeaLevelPressureAvgInches`: Pressure readings
  - `VisibilityAvgMiles`: Visibility metrics
  - `WindAvgMPH`, `WindGustMPH`: Wind speed details
  - `PrecipitationSumInches`: Target variable (Rainfall amount)

## 🛠️ Tech Stack
- **Python**: Core programming language
- **Pandas**: Data manipulation and preprocessing
- **NumPy**: Numerical operations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-Learn**: Machine Learning model (Linear Regression)

## 🚀 Workflow
1. **Data Loading**: Reading the dataset using Pandas.
2. **Data Cleaning**:
   - Dropping unnecessary columns (e.g., `Events`, `Date`, `SeaLevelPressureLowInches`).
   - Handling non-numeric values (replacing 'T' with 0.0, '-' with 0.0).
3. **Exploratory Data Analysis (EDA)**: Visualizing trends and correlations.
4. **Model Training**: Implementing Linear Regression to predict precipitation.
5. **Visualization**: Plotting the precipitation trend graph.

## 📊 Results
The project successfully implements a Linear Regression model to visualize and predict rainfall trends based on historical weather attributes.

## 🔧 How to Run
1. Clone the repository.
2. Navigate to the project directory.
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "rainfall-prediction-project.ipynb"
   ```

---
<div align="center">
  <b>Developed with ❤️ by Harsh Choudhary</b>
</div>
