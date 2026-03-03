<div align="center">
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=0,0,30&height=250&section=header&text=VeriShield%20Fake%20News%20Detector&fontSize=40&animation=fadeIn&fontAlignY=38&desc=NLP-Powered%20Misinformation%20Detection&descAlignY=55&descAlign=50" width="100%" />

  <br/>

  [![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
  [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
  [![NLP](https://img.shields.io/badge/NLP-Misinformation%20Detection-blue?style=for-the-badge)](https://en.wikipedia.org/wiki/Natural_language_processing)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)

---

## 🎯 Overview

Misinformation poses a significant threat to society. This project builds a **machine learning system** to automatically classify news articles as real or fake, helping combat the spread of false information.

### 🌟 Key Objectives
- 🔍 Detect fake news accurately
- 📊 Analyze text patterns
- 🌐 Web interface for easy use
- 💾 Pre-trained models ready to use

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Detection** | Classify news as real/fake |
| 📊 **Confidence** | Probability scores |
| 🌐 **Web App** | Streamlit interface |
| 📝 **Text Analysis** | NLP feature extraction |
| 💾 **Saved Models** | Pre-trained & ready |

---

## 📊 Dataset

| File | Description | Size |
|------|-------------|------|
| `Fake.csv` | Fake news articles | ~63 MB |
| `True.csv` | Real news articles | ~54 MB |

### Combined Dataset
- 📰 **Total Articles**: 40,000+
- 📊 **Balanced Classes**: ~50% each
- 📝 **Features**: Title, Text, Subject, Date

---

## 📁 Project Structure

```
Fake News Detector/
├── Fake_News_detection.ipynb  # Model development
├── app.py                      # Streamlit web app
├── pac_model.pkl               # Trained PA Classifier
├── tfidf_vectorizer.pkl        # TF-IDF vectorizer
├── Fake.csv                    # Fake news dataset
├── True.csv                    # Real news dataset
└── requirements.txt            # Dependencies
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git

# Navigate to project directory
cd "Machine_Learning_Projects/Fake News Detector"

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

### Run Web App
```bash
streamlit run app.py
```

### Run Notebook
```bash
jupyter notebook Fake_News_detection.ipynb
```

---

## 🧠 Model Details

### Algorithm
The project uses **Passive Aggressive Classifier** with **TF-IDF** vectorization.

### Pipeline
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Input Text  │ --> │   TF-IDF     │ --> │     PAC      │
│              │     │  Vectorizer  │     │  Classifier  │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                          ┌──────▼──────┐
                                          │  Prediction │
                                          │  Real/Fake  │
                                          └─────────────┘
```

### Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | ~93%+ |
| **Precision** | High |
| **Recall** | High |
| **F1-Score** | High |

---

## 🔍 How to Use the Web App

1. 📝 Enter or paste news article text
2. 🔘 Click "Check" button
3. ✅ See prediction result
4. 📊 View confidence score

---

## ⚠️ Limitations

- Works best on English news articles
- Training data from specific time period
- Should be used as one verification tool among many

---

<div align="center">
  <h3>⭐ If you found this project useful, please consider giving it a star!</h3>
  <p>Made with ❤️ by <a href="https://github.com/HarshChoudhary2003">Harsh Choudhary</a></p>
  
  <img src="https://capsule-render.vercel.app/render?type=waving&color=gradient&customColorList=0,0,30&height=100&section=footer" width="100%" />
</div>
