<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=2,22,25&height=250&section=header&text=Spam%20Detection%20TensorFlow&fontSize=40&animation=fadeIn&fontAlignY=38&desc=Deep%20Learning%20NLP%20Engine&descAlignY=55&descAlign=50" width="100%" />

  <br/>
  
  [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
  [![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
  [![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)](https://github.com/HarshChoudhary2003)

</div>

---

## 🎯 Overview
Email spam filtering is a mission-critical NLP application. This project builds a high-performance **Deep Learning Neural Network** using TensorFlow to autonomously classify communications with extreme accuracy.

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
  <h3>⭐ If you found this NLP project useful, please consider giving it a star!</h3>
  <p>Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a></p>
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=2,22,25&height=100&section=footer" width="100%" />
</div>
