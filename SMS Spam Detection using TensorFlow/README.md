<div align="center">

# 📱 SMS Spam Detection Using TensorFlow

### *Deep Learning for Text Message Classification*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-NLP-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?style=flat-square" />

---

*Detect spam SMS messages using deep learning to protect users from unwanted texts.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Installation](#-installation)
- [Results](#-results)

---

## 🎯 Overview

SMS spam is a growing concern affecting mobile users globally. This project implements a **TensorFlow-based deep learning model** to classify text messages as spam or legitimate, helping protect users from phishing and scams.

### 🌟 Key Benefits
- 📱 Protect mobile users
- 🔒 Filter unwanted messages
- 🤖 Automated classification
- ⚡ Real-time detection

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `spam.csv` |
| **Size** | ~498 KB |
| **Records** | 5,572 messages |
| **Classes** | Ham (legitimate), Spam |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📝 **Text Preprocessing** | Clean and normalize SMS |
| 🧠 **Deep Learning** | TensorFlow neural network |
| 📊 **High Accuracy** | Reliable classification |
| 🔍 **Pattern Detection** | Learn spam characteristics |

---

## 🔬 Spam Patterns

Common spam indicators detected:
- 💰 "Free", "Win", "Prize"
- 📞 "Call now", "Limited time"
- 🔗 Suspicious URLs
- ⚠️ Urgent language

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/SMS Spam Detection using TensorFlow"

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

### Classification Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | ~97%+ |
| **Precision** | High |
| **Recall** | High |

### Example Predictions
```
📱 "You have won a lottery!" → 🚫 SPAM
📱 "See you at dinner tonight" → ✅ HAM
📱 "Click to claim your prize" → 🚫 SPAM
📱 "Meeting rescheduled to 4pm" → ✅ HAM
```

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
