<div align="center">

# 🛒 Flipkart Reviews Sentiment Analysis

### *NLP-Powered Customer Feedback Analysis*

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=for-the-badge)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

<img src="https://img.shields.io/badge/Status-Complete-success?style=flat-square" />
<img src="https://img.shields.io/badge/Domain-NLP-blue?style=flat-square" />
<img src="https://img.shields.io/badge/Industry-E--Commerce-orange?style=flat-square" />

---

*Analyze customer sentiments from Flipkart product reviews to extract actionable business insights.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [NLP Pipeline](#-nlp-pipeline)
- [Features](#-features)
- [Installation](#-installation)
- [Results](#-results)

---

## 🎯 Overview

This project applies **Natural Language Processing (NLP)** techniques to analyze customer reviews from Flipkart. By understanding customer sentiments, businesses can:

- 📊 Track product perception over time
- 🔍 Identify common pain points
- 💡 Discover improvement opportunities
- 🌟 Enhance customer experience

---

## 📊 Dataset

| Attribute | Description |
|-----------|-------------|
| **File** | `flipkart_data.csv` |
| **Size** | ~1 MB |
| **Content** | Product reviews and ratings |

---

## 🔬 NLP Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Raw Text    │ --> │   Cleaning   │ --> │ Tokenization │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│   Results    │ <-- │  Sentiment   │ <-- │ Vectorization│
│              │     │  Analysis    │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Processing Steps
1. **Text Cleaning** - Remove noise, special characters
2. **Tokenization** - Split into words
3. **Stopword Removal** - Filter common words
4. **Stemming/Lemmatization** - Normalize words
5. **Vectorization** - TF-IDF / Bag of Words
6. **Sentiment Classification** - Positive/Negative/Neutral

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📝 **Text Preprocessing** | Complete NLP pipeline |
| 😊 **Sentiment Scoring** | Polarity detection |
| 📊 **Visualization** | Word clouds & charts |
| 🔍 **Trend Analysis** | Sentiment over time |

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Flipkart Reviews Sentiment Analysis"

# Install dependencies
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud jupyter
```

---

## 💻 Usage

```bash
# Launch Jupyter Notebook
jupyter notebook main.ipynb
```

---

## 📈 Results

| Sentiment | Description |
|-----------|-------------|
| 😊 **Positive** | Satisfied customers |
| 😐 **Neutral** | Mixed feedback |
| 😞 **Negative** | Areas for improvement |

### Sample Insights
- 🌟 Top positive keywords
- ⚠️ Common complaint themes
- 📈 Sentiment distribution

---

<div align="center">

### ⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)

</div>
