<div align="center">

# 📧 Detecting Spam Emails Using TensorFlow

### *Deep Learning for Email Classification*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-NLP-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Deep%20Learning-Neural%20Network-purple?style=flat-square" />

---

*Classify emails as spam or ham using deep learning and natural language processing.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Results](#-results)

---

## 🎯 Overview

Email spam detection is a critical application of NLP and machine learning. This project uses **TensorFlow** to build a deep learning model that accurately classifies emails as spam or legitimate (ham).

### 🌟 Key Features
- 🧠 Deep learning with TensorFlow
- 📝 Text preprocessing pipeline
- 📊 High accuracy classification
- 🔒 Protect inboxes from spam

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `spam_ham_dataset.csv` |
| **Size** | ~5.5 MB |
| **Classes** | Spam, Ham |
| **Content** | Email text and labels |

---

## 🧠 Architecture

### NLP Pipeline
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Email   │ --> │  Tokenize    │ --> │   Padding    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│  Prediction  │ <-- │    Dense     │ <-- │  Embedding   │
│  Spam/Ham    │     │   Layers     │     │    Layer     │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Model Structure
| Layer | Details |
|-------|---------|
| 📝 **Embedding** | Word vectors |
| 🔄 **LSTM/Dense** | Feature learning |
| 🚫 **Dropout** | Regularization |
| 📊 **Output** | Binary classification |

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Detecting Spam Emails Using Tensorflow"

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

### Model Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 95%+ |
| **Precision** | High |
| **Recall** | High |
| **F1-Score** | High |

### Classification Examples
```
📧 "Congratulations! You've won $1000000!" → 🚫 SPAM
📧 "Meeting tomorrow at 3pm" → ✅ HAM
📧 "Click here for free iPhone" → 🚫 SPAM
📧 "Project deadline reminder" → ✅ HAM
```

---

## 💡 How It Works

1. **Tokenization** - Convert text to sequences
2. **Embedding** - Map words to dense vectors
3. **Learning** - Neural network learns patterns
4. **Classification** - Predict spam probability

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
