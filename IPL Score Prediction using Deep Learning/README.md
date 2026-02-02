# IPL Score Prediction using Deep Learning 🏏

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.0%2B-red.svg)](https://keras.io/)

This project implements a Deep Learning model to predict the final score of an IPL match based on the current state of the game. Using historical IPL data, the model learns complex patterns to provide accurate score estimations.

## 📌 Project Overview

Predicting the score in a cricket match, especially in a fast-paced format like T20 (IPL), is a challenging task due to various influencing factors. This project leverages Artificial Neural Networks (ANN) to analyze game situations and predict the final total.

## 📊 Dataset Description

The dataset `ipl_data.csv` contains ball-by-ball information of IPL matches from 2008 to 2017.

**Key Features:**
- `mid`: Match ID
- `date`: Date of the match
- `venue`: Stadium where the match was played
- `bat_team`: Batting team
- `bowl_team`: Bowling team
- `batsman`: Striker
- `bowler`: Bowler
- `runs`: Current runs
- `wickets`: Current wickets lost
- `overs`: Current overs completed
- `runs_last_5`: Runs scored in the last 5 overs
- `wickets_last_5`: Wickets lost in the last 5 overs
- `striker`: Runs scored by the striker
- `non-striker`: Runs scored by the non-striker
- `total`: Total runs scored (Target Variable)

## 🛠️ Technologies Used

- **Data Manipulation:** `Pandas`, `NumPy`
- **Data Visualization:** `Matplotlib`, `Seaborn`
- **Preprocessing:** `Scikit-learn` (Label Encoding, StandardScaler)
- **Deep Learning Framework:** `TensorFlow`, `Keras`

## 🧠 Model Architecture

The core of this project is a Deep Neural Network built using Keras:
- **Input Layer:** Corresponding to the selected features.
- **Hidden Layers:** Multiple dense layers with `ReLU` activation.
- **Output Layer:** A single neuron with linear activation for regression.
- **Optimizer:** `Adam`
- **Loss Function:** `Mean Squared Error` (MSE) / `Mean Absolute Error` (MAE)

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
   Open `main.ipynb` in your favorite Jupyter environment (Jupyter Lab, VS Code, Google Colab) and run the cells sequentially.

## 📈 Results

The model achieves significant accuracy in predicting the final scores, allowing for real-time score updates and predictions during live matches.

## 🤝 Contributing

Contributions are welcome! If you have any ideas to improve the model or add new features, feel free to open a Pull Request.

---
Developed with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)
