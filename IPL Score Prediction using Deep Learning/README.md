<div align="center">

# IPL Score Prediction using Deep Learning 🏏

### *Deep Learning for Real-Time Cricket Analytics*

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.0%2B-red.svg)](https://keras.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Sport-IPL-004BA0?style=flat-square" />
<img src="https://img.shields.io/badge/Deep%20Learning-ANN-blue?style=flat-square" />

---

*Predict the final score of an IPL match based on real-time game state using Artificial Neural Networks (ANN).*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset Description](#-dataset-description)
- [Technologies Used](#-technologies-used)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Results](#-results)

---

## 📌 Project Overview

Predicting the score in a cricket match, especially in a fast-paced format like T20 (IPL), is a challenging task due to various influencing factors such as wickets lost, run rate, and stadium dimensions. This project leverages **Deep Learning (ANN)** to analyze historical ball-by-ball data and provide accurate final score estimations.

### 🌟 Key Features
- 🏏 **Real-time Prediction** - Calculate projected totals based on current overs/wickets.
- 📊 **Historical Analysis** - Trained on a decade of IPL data (2008-2017).
- 🧠 **Complex Pattern Recognition** - Captures non-linear relationships between game variables.

---

## 📊 Dataset Description

The dataset `ipl_data.csv` contains comprehensive ball-by-ball information of IPL matches.

| Feature | Description |
|---------|-------------|
| `mid` | Unique Match ID |
| `venue` | Stadium name |
| `bat_team` | Current batting team |
| `bowl_team` | Current bowling team |
| `runs` | Current runs scored |
| `wickets` | Current wickets lost |
| `overs` | Current overs completed |
| `runs_last_5` | Runs scored in the previous 5 overs |
| `wickets_last_5` | Wickets lost in the previous 5 overs |
| **`total`** | **Target Variable: Final score** |

---

## 🧠 Model Architecture

The system uses a Deep Feed-Forward Neural Network:
- **Input Layer:** Normalized features representing the current match state.
- **Hidden Layers:** Multiple dense layers with `ReLU` activation for learning deep representations.
- **Output Layer:** A single neuron with linear activation for regression (score prediction).
- **Optimization:** `Adam` optimizer with `Mean Squared Error` (MSE) loss function.

---

## 🛠️ Technologies Used

- **Frameworks:** `TensorFlow`, `Keras`
- **Data:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **ML Utilities:** `Scikit-learn` (StandardScaler, LabelEncoder)

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "IPL Score Prediction using Deep Learning"
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
   ```

3. **Run the Notebook:**
   Open `main.ipynb` and execute the cells to train and evaluate the model.

---

## 📈 Results

The model achieves high precision in predicting final totals, particularly in the later stages of the first innings. It provides a robust alternative to traditional "current run rate" projections.

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
