<div align="center">

# 📝 Classification of Text Documents using Naive Bayes

### *NLP Text Classification with Machine Learning*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)](https://www.nltk.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-NLP-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Algorithm-Naive%20Bayes-purple?style=flat-square" />

---

*Classify text documents into categories using the Naive Bayes algorithm and NLP techniques.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Algorithm](#-algorithm)
- [Pipeline](#-pipeline)
- [Installation](#-installation)
- [Results](#-results)

---

## 🎯 Overview

Text classification is a fundamental NLP task. This project implements the **Naive Bayes classifier** to automatically categorize text documents, demonstrating the power of probabilistic machine learning in NLP.

### 🌟 Use Cases
- 📧 Email categorization
- 📰 News article classification
- 🏷️ Topic labeling
- 📋 Document organization

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `synthetic_text_data.csv` |
| **Size** | ~8 KB |
| **Content** | Text documents with labels |

---

## 🧠 Algorithm

### Why Naive Bayes?

```
P(Category | Document) ∝ P(Document | Category) × P(Category)
```

| Advantage | Description |
|-----------|-------------|
| ⚡ **Fast** | Quick training and prediction |
| 💾 **Efficient** | Low memory requirement |
| 📊 **Effective** | Works well for text |
| 🎯 **Simple** | Easy to implement |

### Types Used
- 📊 **Multinomial NB** - Word counts
- 🔢 **Bernoulli NB** - Binary features
- 📈 **Gaussian NB** - Continuous features

---

## 🔬 Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Raw Text   │ --> │  Preprocess  │ --> │  Tokenize    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│   Predict    │ <-- │  Train NB    │ <-- │  Vectorize   │
└──────────────┘     └──────────────┘     │  (TF-IDF)    │
                                          └──────────────┘
```

### Steps
1. **Text Preprocessing** - Clean & normalize
2. **Tokenization** - Split into words
3. **Vectorization** - TF-IDF transformation
4. **Training** - Fit Naive Bayes
5. **Prediction** - Classify new documents

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Classification of Text Documents using Naive Bayes"

# Install dependencies
pip install pandas numpy scikit-learn nltk matplotlib seaborn jupyter
```

---

## 💻 Usage

```bash
# Launch Jupyter Notebook
jupyter notebook main.ipynb
```

---

## 📈 Results

| Metric | Score |
|--------|-------|
| **Accuracy** | Calculated in notebook |
| **Precision** | Calculated in notebook |
| **Recall** | Calculated in notebook |
| **F1-Score** | Calculated in notebook |

### Confusion Matrix
Visual representation of classification performance available in notebook.

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
