# 📘 Facebook Sentiment Analysis using Python

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-2E7D32?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

## 📌 Overview

This project performs **Sentiment Analysis** on text data (simulating Facebook posts or Kindle reviews) using **Python** and the **NLTK** library. It utilizes the **VADER (Valence Aware Dictionary and sEntiment Reasoner)** sentiment analyzer to determine if a piece of text is positive, negative, or neutral.

## 🚀 Features

- **Text Preprocessing**:
  - Tokenization (Sentence & Word)
  - Stemming (PorterStemmer)
  - Lemmatization (WordNetLemmatizer)
  - Part-of-Speech (POS) Tagging
- **Sentiment Analysis**:
  - Uses `SentimentIntensityAnalyzer` from `nltk.sentiment.vader`
  - Calculates Polarity Scores: Compound, Positive, Negative, Neutral

## 🛠️ Tech Stack

- **Python**: Core programming language
- **NLTK (Natural Language Toolkit)**: For NLP tasks (tokenization, stemming, lemmatization, sentiment analysis)
- **Pandas & NumPy**: For data handling (if expanded)
- **Matplotlib**: For visualization (potential future use)

## 📂 Project Structure

```
Facebook Sentiment Analysis using python/
├── main.ipynb       # Jupyter Notebook containing the code
├── kindle.txt       # Sample text file for analysis
└── README.md        # Project documentation
```

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Machine_Learning_Projects/Facebook Sentiment Analysis using python"
   ```

2. **Install dependencies**:
   ```bash
   pip install nltk pandas numpy matplotlib
   ```

3. **Run the Notebook**:
   Open `main.ipynb` in Jupyter Notebook or VS Code and run the cells.
   
   The notebook will automatically download necessary NLTK data:
   - `vader_lexicon`
   - `punkt`
   - `wordnet`
   - `averaged_perceptron_tagger`
   - `omw-1.4`

## 📊 Results

The model outputs sentiment scores for each text entry:

```
I really loved this book!
compound: 0.8012, neg: 0.0, neu: 0.386, pos: 0.614
```

- **Compound score** > 0.05: Positive
- **Compound score** < -0.05: Negative
- Otherwise: Neutral

## 🤝 Contributing

Feel free to fork this repository and submit pull requests to enhance the analysis or add visualization features!
