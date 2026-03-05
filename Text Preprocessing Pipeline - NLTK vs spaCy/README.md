<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=10,30,50&height=280&section=header&text=NLTK%20vs%20spaCy%20Pipeline&fontSize=45&animation=fadeIn&fontAlignY=38&desc=Comparative%20Study%20of%20NLP%20Preprocessing%20%26%20ML%20Performance&descAlignY=55&descAlign=50" width="100%" />

  <br/>

  [![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![NLTK](https://img.shields.io/badge/NLTK-NLP-orange?style=for-the-badge)](https://www.nltk.org/)
  [![spaCy](https://img.shields.io/badge/spaCy-NLP-092E20?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
  [![ML](https://img.shields.io/badge/Machine%20Learning-Comparison-green?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

</div>

---

## 🎯 Project Overview

This project provides a comprehensive comparison between two of the most popular Natural Language Processing (NLP) libraries: **NLTK** and **spaCy**. It covers the entire preprocessing pipeline, benchmarks execution speeds, and evaluates their impact on a Machine Learning (Sentiment Analysis) task.

### 🔍 Key Components
- **Advanced Text Cleaning**: Handling URLs, emojis, hashtags, and contractions.
- **Deep NLP Analysis**: Tokenization, POS Tagging, Lemmatization, and Named Entity Recognition (NER).
- **Benchmarking**: Execution time comparison for NLTK vs spaCy on large-scale tweet datasets.
- **ML Evaluation**: Logistic Regression baseline using TF-IDF features to measure preprocessing effectiveness.
- **Visualizations**: Word Clouds, N-Gram analysis, and Dependency Parse Trees.

---

## 📊 Observations & Insights

| Feature | NLTK | spaCy |
|:---|:---|:---|
| **Philosophy** | Tool-kit (Modular) | Production-ready (Pipeline) |
| **Speed** | ⚡ Fast (Standard tasks) | 🚀 Ultra Fast (Large datasets) |
| **Lemmatization** | WordNet (Lexical) | Dependency-based (Contextual) |
| **NER** | Basic | 🌟 State-of-the-Art |
| **Ease of Use** | Flexibility in steps | Streamlined API |

> [!TIP]
> **Conclusion:** While NLTK offers more granular control for research purposes, spaCy is significantly more efficient for building robust production pipelines.

---

## 🛠️ Tech Stack

- **NLP**: `nltk`, `spacy`
- **Machine Learning**: `scikit-learn`
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `wordcloud`
- **Utilities**: `emoji`, `contractions`, `re`

---

## 🚀 Quick Start

### 1. Clone & Navigate
```bash
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
cd "Text Preprocessing Pipeline - NLTK vs spaCy"
```

### 2. Install Dependencies
```bash
pip install nltk spacy matplotlib seaborn wordcloud contractions emoji scikit-learn
python -m spacy download en_core_web_sm
```

### 3. Run the Analysis
Open `Main.ipynb` in your Jupyter environment or run:
```bash
jupyter notebook Main.ipynb
```

---

## 📈 Visualizing Results

The notebook includes:
- **Bi-gram Analysis**: Discovering the most frequent local phrase patterns.
- **Sentiment Distribution**: Understanding the balance of the dataset.
- **Confusion Matrix**: Visualizing the precision/recall of our ML model.
- **Dependency Trees**: Understanding sentence structure via spaCy's `displacy`.

---

<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=10,30,50&height=120&section=footer" width="100%" />
</div>
