# 🩺 Cancer Cell Classification using Scikit-learn

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 📌 Overview

This project builds a **Machine Learning model** to classify cancer cells as either **malignant** (harmful) or **benign** (non-harmful) based on various features extracted from digitized images of a fine needle aspirate (FNA) of a breast mass. It utilizes the **Gaussian Naive Bayes** algorithm from the **Scikit-learn** library.

## 🚀 Features

- **Data Loading**: Uses the `load_breast_cancer` dataset from Scikit-learn.
- **Exploratory Data Analysis (EDA)**:
  - Data structure overview (`data.info()`, `data.describe()`)
  - Target class distribution visualization (Pie Chart)
- **Model Training**:
  - Splits data into training and testing sets.
  - Trains a **Gaussian Naive Bayes (GaussianNB)** classifier.
- **Evaluation**:
  - Calculates the accuracy score of the model.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **Scikit-Learn**: For dataset loading, model training (GaussianNB), and evaluation metrics.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.

## 📂 Project Structure

```
Cancer cell classification using Scikit-learn/
├── main.ipynb       # Jupyter Notebook containing the code
└── README.md        # Project documentation
```

## ⚙️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Machine_Learning_Projects/Cancer cell classification using Scikit-learn"
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas matplotlib scikit-learn
   ```

3. **Run the Notebook**:
   Open `main.ipynb` in Jupyter Notebook or VS Code and run the cells.

## 📊 Dataset Details

The dataset features describe characteristics of the cell nuclei present in the image.
- **Classes**: Malignant, Benign
- **Samples**: 569
- **Dimensionality**: 30 features (radius, texture, perimeter, area, smoothness, etc.)

## 🤝 Contributing

Contributions are welcome! Feel free to improve the model accuracy by trying different algorithms (e.g., SVM, Random Forest) or adding more visualizations.
