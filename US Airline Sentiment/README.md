
# ✈️ US Airline Sentiment Analysis AI

![Sentiment Analysis](https://komarev.com/ghpvc/?username=HarshChoudhary2003&label=PROJECT%20VIEWS&color=0e75b6&style=flat)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25.0+-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/ML-Sentiment%20Analysis-orange.svg)

An end-to-end Machine Learning pipeline designed to analyze and predict customer sentiment from US Airline tweets. This project leverages natural language processing (NLP) techniques, robust machine learning, and an interactive dashboard for actionable insights.

---

## 🚀 Features

- **Advanced NLP Pipeline**: Comparative analysis using NLTK and SpaCy for tokenization, lemmatization, and POS tagging.
- **Sentiment Insights**: Deep exploratory analysis of negative sentiment triggers (delays, baggage, customer service).
- **Interactive Dashboard**: A premium Streamlit UI to visualize datasets and real-time predictions.
- **ML Engine**: Logistic Regression model trained on Tfidf features with 80%+ accuracy.
- **Fast Inference**: Pre-trained model artifacts for instantaneous sentiment prediction.

---

## 🛠️ Tech Stack

- **Preprocessing**: `NLTK`, `SpaCy`, `Regex`, `Contractions`
- **Machine Learning**: `Scikit-Learn`, `TfidfVectorizer`, `Joblib`
- **Visualization**: `Plotly`, `WordCloud`, `Seaborn`
- **Web App**: `Streamlit`

---

## 📊 Dataset Overview

The dataset contains ~14,600 tweets about major US airlines (Virgin America, United, Southwest, Delta, US Airways, American).
- **Labels**: Positive, Neutral, Negative.
- **Additional Data**: Negative reasons, airline name, tweet location.

---

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "US Airline Sentiment"
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   ```bash
   python ml_model.py
   ```

4. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## 🍎 Visual Showcase

### Interactive Predictor
> Analyze individual tweets with confidence scoring.

### Data Dashboard
> View airline-specific sentiment distributions and complaint themes.

---

## 🤝 Contributing
Feel free to fork this project and submit PRs for improvements (e.g., adding BERT models, expanding airline coverage).

---
*Created with ❤️ by Antigravity AI*
