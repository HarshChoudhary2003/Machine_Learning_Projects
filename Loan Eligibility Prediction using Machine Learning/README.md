# 🏦 Loan Eligibility Prediction using Machine Learning

<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=2,22,25&height=250&section=header&text=Loan%20Eligibility%20Prediction&fontSize=40&animation=fadeIn&fontAlignY=38&desc=Machine%20Learning%20Model%20for%20Loan%20Approval&descAlignY=55&descAlign=50" width="100%" />

  <br/>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
  [![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
  [![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-blue?style=for-the-badge&logo=python&logoColor=white)](https://imbalanced-learn.org/)

</div>

---

## 📖 Table of Contents

- [🎯 Overview](#-overview)
- [📊 Dataset](#-dataset)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚙️ Methodology](#️-methodology)
- [🚀 How to Run](#-how-to-run)
- [📈 Results](#-results)
- [🤝 Contributing](#-contributing)

---

## 🎯 Overview

Loan approval is a critical process for banking institutions. This project aims to automate the loan eligibility process (real-time) based on customer details provided while filling out the online application form.

The system uses **Machine Learning** to predict whether a loan will be approved or not, helping to identify eligible candidates and reduce manual work.

---

## 📊 Dataset

The dataset used for this project contains details about the loan applicants.

**Key Features:**
- **Gender**: Male/Female
- **Married**: Applicant married (Y/N)
- **Education**: Applicant Education (Graduate/Under Graduate)
- **Self_Employed**: Self-employed (Y/N)
- **ApplicantIncome**: Applicant income
- **CoapplicantIncome**: Coapplicant income
- **LoanAmount**: Loan amount in thousands
- **Loan_Amount_Term**: Term of loan in months
- **Credit_History**: Credit history meets guidelines
- **Property_Area**: Urban/Semi Urban/Rural
- **Loan_Status**: (Target) Loan approved (Y/N)

---

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**:
    - `pandas` & `numpy` for data manipulation
    - `matplotlib` & `seaborn` for visualization
    - `scikit-learn` for model building and evaluation
    - `imbalanced-learn` for handling class imbalance

---

## ⚙️ Methodology

1.  **Data Preprocessing**: 
    - Handling missing values.
    - Encoding categorical variables using `LabelEncoder`.
    - Scaling numerical features using `StandardScaler`.
2.  **Exploratory Data Analysis (EDA)**: Visualizing relationships between features and the target variable.
3.  **Handling Imbalance**: Using `RandomOverSampler` to balance the dataset (since loan approvals might be skewed).
4.  **Model Selection**: Using Support Vector Classifier (`SVC`) for classification.
5.  **Evaluation**: Assessing performance using accuracy score and classification reports.

---

## 🚀 How to Run

1.  **Clone the repository** (if not already done):
    ```bash
    git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
    cd "Machine_Learning_Projects/Loan Eligibility Prediction using Machine Learning"
    ```

2.  **Install dependencies**:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
    ```

3.  **Run the Notebook**:
    You can run the `main.ipynb` file using Jupyter Notebook or JupyterLab.
    ```bash
    jupyter notebook main.ipynb
    ```

---

## 📈 Results

The model learns patterns from historical data to predict the likelihood of loan approval. Accuracy and classification metrics are calculated to evaluate the model's effectiveness.

*(Run the notebook to see the specific accuracy scores and confusion matrix)*

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model or adding new features, feel free to open an issue or submit a pull request.

<div align="center">
  <br/>
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=2,22,25&height=100&section=footer" width="100%" />
</div>
