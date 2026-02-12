# 🩺 Tumor Detection Project

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
</div>

## 📌 Overview
This project focuses on detecting tumors using Machine Learning techniques. The goal is to classify tumors as **Malignant** or **Benign** based on various medical attributes.

## 📂 Dataset
The dataset used is `Tumor_Detection.csv`, which contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
- **Target Variable:** `diagnosis` (M = malignant, B = benign)
- **Features:** Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave points, Symmetry, Fractal dimension.

## 🚀 Workflow
1. **Data Preprocessing**: Handling missing values, encoding categorical variables.
2. **EDA**: Visualizing feature distributions and correlations.
3. **Model Training**: Utilizing classification algorithms (e.g., Logistic Regression, Random Forest).
4. **Evaluation**: Measuring accuracy, precision, recall, and F1-score.

## 🔧 How to Run
1. Clone the repository.
2. Navigate to the project directory:
   ```bash
   cd Tumor_Detection-project
   ```
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
4. Run the notebook:
   ```bash
   jupyter notebook Tumor_Detection-project.ipynb
   ```
