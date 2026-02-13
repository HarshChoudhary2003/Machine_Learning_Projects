# 🔢 Recognizing Handwritten Digits using Scikit-Learn

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

<div align="center">
  <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_digits_classification_001.png" alt="Handwritten Digits" width="600"/>
</div>

## 📄 Overview

This project demonstrates how to build a **Handwritten Digit Recognition System** using **Scikit-Learn**. Instead of using deep learning frameworks like TensorFlow or PyTorch, this project utilizes Scikit-Learn's **Multi-Layer Perceptron (MLP) Classifier** to achieve high accuracy on the digits dataset.

The project covers data loading, visualization, preprocessing, model training, and evaluation, providing a clear example of using neural networks within the Scikit-Learn ecosystem.

## 🔑 Key Features

- **Data Visualization**: Visualizing the pixel data of handwritten digits using Matplotlib.
- **MLP Classifier**: Implementing a Multi-Layer Perceptron (Neural Network) classifier using `sklearn.neural_network`.
- **Model Training**: Detailed training process with loss curve monitoring.
- **Evaluation**: Assessing the model's performance on unseen test data.

## 🛠️ Technologies Used

- **Python 3.8+**: The core programming language.
- **Scikit-Learn**: For the machine learning algorithm (MLPClassifier) and dataset.
- **Matplotlib**: For plotting and visualizing the digits.
- **NumPy**: For numerical operations and array manipulation.

## 📂 Project Structure

```
Recognizing HandWritten Digits in Scikit Learn/
├── main.ipynb   # Jupyter Notebook containing the code
└── README.md    # Project Documentation
```

## 🚀 Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
    cd "Machine_Learning_Projects/Recognizing HandWritten Digits in Scikit Learn"
    ```

2.  **Install dependencies**:
    ```bash
    pip install scikit-learn matplotlib numpy notebook
    ```

3.  **Run the Notebook**:
    ```bash
    jupyter notebook main.ipynb
    ```

## 📊 Results

The MLP Classifier effectively learns to distinguish between the handwritten digits (0-9) by training on the pixel values. The notebook demonstrates the convergence of the loss function over iterations.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the model accuracy or adding new visualizations, feel free to fork the repo and submit a pull request.

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a>
</div>
