# Predicting Stock Price Direction using Support Vector Machines

## 📌 Project Overview
This project applies **Machine Learning (Support Vector Machines - SVM)** to predict the direction of stock prices. Using historical stock data (Reliance Industries), the model classifies whether the stock price will go **UP (1)** or **DOWN (0)** the next day.

The project explores different **SVM kernels** (Linear, Polynomial, RBF, Sigmoid) and implements a simple trading strategy to backtest the model's performance.

## 📂 Dataset
The dataset used is `RELIANCE.csv`, which contains historical stock market data with the following columns:
- **Date**: The date of the record.
- **Open**: Opening price of the stock.
- **High**: Highest price reached during the day.
- **Low**: Lowest price reached during the day.
- **Close**: Closing price of the stock.
- **Adj Close**: Adjusted closing price.
- **Volume**: Number of shares traded.

## 🛠️ Tech Stack
- **Python 3.8+**
- **Libraries**:
  - `pandas` (Data Manipulation)
  - `numpy` (Numerical Computations)
  - `matplotlib` (Data Visualization)
  - `scikit-learn` (Machine Learning - SVM, Metrics)

## 🏗️ Methodology
1. **Data Preprocessing**:
   - Loaded the dataset and set `Date` as the index.
   - Created predictor variables:
     - `Open-Close`: The difference between Open and Close prices.
     - `High-Low`: The difference between High and Low prices.
   - Defined the target variable `y`: **1** if the next day's price > current day's price, else **0**.

2. **Model Training**:
   - Split data into **Training (80%)** and **Testing (20%)** sets.
   - Trained a **Support Vector Classifier (SVC)**.
   - Experimented with different kernels:
     - **Linear**
     - **Polynomial (Degree 3)**
     - **RBF (Radial Basis Function)**
     - **Sigmoid**

3. **Evaluation**:
   - Evaluated the model using **Accuracy Score**.
   - Implemented a trading strategy based on model predictions to calculate potential returns.

## 📊 Results
The model's accuracy was evaluated across different kernels. (Refer to the notebook for specific accuracy metrics).

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   ```
2. Navigate to the project directory:
   ```bash
   cd "Machine_Learning_Projects/Predicting Stock Price Direction using Support Vector Machines"
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

## 📜 License
This project is licensed under the MIT License.
