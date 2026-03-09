# 🧠 Word Embeddings & Emotion Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Natural Language Processing](https://img.shields.io/badge/NLP-Emotion%20Recognition-green.svg)](https://en.wikipedia.org/wiki/Emotion_recognition)

A comprehensive exploration of **Word Embeddings** (Word2Vec, FastText, GloVe) and their application in **Multi-label Emotion Recognition** using Deep Learning (BiLSTM + Attention).

---

## 🚀 Overview

This project implements a complete NLP pipeline for:
1.  **Custom Embedding Training**: Training Word2Vec and FastText models on a provided corpus.
2.  **Embedding Analysis**: Visualizing semantic relationships using t-SNE and Cosine Similarity Heatmaps.
3.  **Emotion Classification**: Building a state-of-the-art **BiLSTM with Attention** model to classify emotions in the **GoEmotions** dataset.
4.  **Baseline Comparison**: Benchmarking deep learning models against traditional Machine Learning (Logistic Regression, Random Forest).

---

## 🏗️ Architecture

The project is structured into modular components:

-   `run_project.py`: Main entry point for the full execution flow.
-   `src/emotion_classifier.py`: Deep learning models (BiLSTM, CNN, Attention).
-   `src/word2vec_training.py`: Training logic for Word2Vec and FastText.
-   `src/embedding_analysis.py`: Visualization tools (t-SNE, Heatmaps).
-   `src/glove_loader.py`: Utility to load pre-trained GloVe vectors.
-   `src/baseline_models.py`: Traditional Scikit-Learn baselines.

---

## 📊 Feature Highlights

### 1. Advanced Deep Learning
We utilize a **Bidirectional LSTM with an Attention Mechanism**. The Attention layer allows the model to focus on the most salient words in a sentence (e.g., "happy", "frustrating") to better predict complex emotions.

### 2. Multi-Embedding Support
Compare and contrast different embedding techniques:
-   **Word2Vec**: Context-based word representations.
-   **FastText**: Sub-word information for better OOV (Out-of-Vocabulary) handling.
-   **GloVe**: Pre-trained global word-word co-occurrence statistics.

### 3. Rich Visualizations
The project automatically generates:
-   **t-SNE Plots**: 2D/3D clusters of semantic word groups.
-   **Similarity Heatmaps**: Visual representation of word-word relationships.
-   **Confusion Matrices**: Detailed error analysis for classification.
-   **Learning Curves**: Tracking training progress over time.

---

## 🛠️ Setup & Usage

### Prerequisites
Ensure you have the following installed:
-   Python 3.8+
-   `pip install -r requirements.txt`

### Dataset
Place the `go_emotions_dataset.csv` in the root directory. If missing, the project will use a dummy/sample subset for demonstration.

### Running the Project
```bash
python run_project.py
```

---

## 📈 Example Results

| Model | Accuracy | F1-Macro |
| :--- | :--- | :--- |
| **BiLSTM + Attention** | **~0.48** | **~0.42** |
| Random Forest | ~0.35 | ~0.28 |
| Logistic Regression | ~0.38 | ~0.32 |

*(Note: Results based on subset training for demonstration).*

---

## 🔮 Prediction Demo

You can now predict emotions for custom text:
```python
pred = classifier.predict_single(model, "I am so happy about this project!", emotion_cols)
# Output: {'joy': 0.92, 'excitement': 0.85}
```

---

## 🤝 Contributing
Feel free to fork this repository and submit pull requests for any enhancements!

---

Developed with ❤️ for NLP Enthusiasts.
