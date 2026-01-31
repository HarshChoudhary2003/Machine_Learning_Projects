
<div align="center">

# ❤️ Heart Disease Prediction System

### *Predicting 10-Year Cardiovascular Risk*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)

<img src="https://img.shields.io/badge/Model-Logistic%20Regression-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Accuracy-85%25-green?style=flat-square" />
<img src="https://img.shields.io/badge/Dataset-Framingham-red?style=flat-square" />

---

*A Machine Learning project to predict the 10-year risk of developing Coronary Heart Disease (CHD) using Logistic Regression.*

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Installation](#-installation)
- [Usage](#-usage)

---

## 🎯 Overview

Cardiovascular diseases (CVDs) are the number one cause of death globally. Early detection and management are crucial. This project leverages the famous **Framingham Heart Study** dataset to build a predictive model that estimates the 10-year risk of future coronary heart disease (CHD) for an individual.

We define this as a binary classification problem:
- **0**: No risk of CHD in the next 10 years.
- **1**: Risk of CHD in the next 10 years.

---

## 📊 Dataset

The dataset used is from an ongoing cardiovascular study on residents of the town of Framingham, Massachusetts.

- **Source**: Framingham Heart Study
- **Samples**: ~4,000 records
- **Attributes**: 15 clinical, behavioral, and demographic features.

### Key Attributes:
| Attribute | Description |
|-----------|-------------|
| `male` | Gender (0 = Female, 1 = Male) |
| `age` | Age of the patient |
| `currentSmoker` | Whether the patient is a smoker |
| `cigsPerDay` | Number of cigarettes smoked per day |
| `BPMeds` | Whether on blood pressure medication |
| `prevalentStroke` | History of stroke |
| `prevalentHyp` | History of hypertension |
| `diabetes` | History of diabetes |
| `totChol` | Total cholesterol level |
| `sysBP` | Systolic blood pressure |
| `diaBP` | Diastolic blood pressure |
| `BMI` | Body Mass Index |
| `heartRate` | Heart rate |
| `glucose` | Glucose level |
| **Target** | `TenYearCHD` (10-year risk of coronary heart disease) |

---

## ⚙️ Methodology

1.  **Data Cleaning**: 
    - Handling missing values (dropping rows with Nulls).
    - Removing irrelevant columns (e.g., `education`).
2.  **Exploratory Data Analysis (EDA)**:
    - Visualizing the distribution of the target variable.
    - Analysis of feature correlations.
3.  **Preprocessing**:
    - **Normalization**: Scaling features using `StandardScaler` to ensure all features contribute equally.
    - **Splitting**: Dividing data into training (70%) and testing (30%) sets.
4.  **Modeling**:
    - Implementing **Logistic Regression**, a statistical model suitable for binary classification tasks.

---

## 📈 Model Performance

The Logistic Regression model achieved reliable accuracy on the test set.

| Metric | Result |
|--------|--------|
| **Accuracy** | **~85%** |
| **Precision (Class 0)** | High (0.85) |
| **Recall (Class 0)** | High (0.99) |

*Note: The dataset is imbalanced (fewer positive cases), which is typical for medical datasets. Further improvements could involve techniques like SMOTE or class weighting.*

---

## 🚀 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
    cd "Machine_Learning_Projects/Heart Disease Prediction Using Logistic Regression"
    ```

2.  **Install dependencies**
    You need Python and the following libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

---

## 💻 Usage

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook main.ipynb
    ```
2.  Run the cells sequentially to load data, train the model, and view predictions.

---

<div align="center">

### ⭐ Star this repo if you find it helpful!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
