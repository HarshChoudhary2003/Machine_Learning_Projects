<div align="center">

# ✋ Handwritten Digit Recognition

### *Neural Networks for MNIST Classification*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-Computer%20Vision-purple?style=flat-square" />
<img src="https://img.shields.io/badge/Deep%20Learning-CNN-blue?style=flat-square" />

---

*Classify handwritten digits (0-9) using deep neural networks on the famous MNIST dataset.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Results](#-results)

---

## 🎯 Overview

Handwritten digit recognition is a classic computer vision problem and serves as the "Hello World" of deep learning. This project implements **neural networks** to recognize digits from images with high accuracy.

### 🌟 Learning Outcomes
- 🧠 Deep learning fundamentals
- 🖼️ Image classification
- 📊 Model evaluation
- 🔧 Hyperparameter tuning

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `Train.csv` |
| **Size** | ~77 MB |
| **Images** | 28x28 grayscale |
| **Classes** | 0-9 digits |

### Sample Distribution
```
┌────────────────────────────────────────────┐
│         DIGIT DISTRIBUTION                 │
├────────────────────────────────────────────┤
│  0: ████████ │  5: ████████               │
│  1: █████████│  6: ████████               │
│  2: ████████ │  7: █████████              │
│  3: ████████ │  8: ████████               │
│  4: ████████ │  9: ████████               │
└────────────────────────────────────────────┘
```

---

## 🧠 Model Architecture

### Neural Network Structure
```
┌─────────────────────────────────────────────┐
│            INPUT LAYER (784)                │
│            (28 × 28 flattened)              │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│          HIDDEN LAYER 1 (128)               │
│            ReLU Activation                  │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│          HIDDEN LAYER 2 (64)                │
│            ReLU Activation                  │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│           OUTPUT LAYER (10)                 │
│           Softmax Activation                │
└─────────────────────────────────────────────┘
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Handwritten Digit Recognition using Neural Network"

# Install dependencies
pip install pandas numpy tensorflow keras matplotlib scikit-learn jupyter
```

---

## 💻 Usage

```bash
# Launch Jupyter Notebook
jupyter notebook main.ipynb
```

---

## 📈 Results

### Performance Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | ~98%+ |
| **Loss** | Minimized |

### Confusion Matrix
Visual representation showing prediction accuracy for each digit class.

### Sample Predictions
```
Actual: 7  →  Predicted: 7 ✓
Actual: 2  →  Predicted: 2 ✓
Actual: 1  →  Predicted: 1 ✓
Actual: 0  →  Predicted: 0 ✓
```

---

## 🔧 Model Training

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Loss** | Categorical Crossentropy |
| **Epochs** | 10-20 |
| **Batch Size** | 32 |

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
