# TED Talks Recommendation System 📽️

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-green.svg)](https://www.nltk.org/)

A content-based recommendation system that suggests TED talks based on their textual descriptions and titles using Natural Language Processing (NLP) and vector similarity.

## 📌 Project Overview

With thousands of TED talks available, finding the most relevant content can be overwhelming. This project leverages machine learning to analyze the themes and topics of talks to recommend similar content to viewers based on what they are currently watching.

## 🛠️ Technology Stack

- **Data Processing:** `Pandas`, `NumPy`
- **Natural Language Processing:** `NLTK` (Stopwords, Tokenization)
- **Vectorization:** `Scikit-Learn` (TfidfVectorizer)
- **Similarity Metric:** `Cosine Similarity`
- **Visualization:** `Matplotlib`, `WordCloud`

## 🧠 Methodology

1.  **Data Cleaning:** 
    - Combined `title` and `details` into a single feature.
    - Removed punctuations and common English stopwords.
    - Converted all text to lowercase for consistency.
2.  **Exploratory Data Analysis (EDA):**
    - Analyzed the distribution of talks across years and months.
    - Generated word clouds to visualize prominent themes.
3.  **Feature Engineering:**
    - Applied **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to transform text data into numerical features, capturing the importance of words within the context of the dataset.
4.  **Recommendation Engine:**
    - Calculated the **Cosine Similarity** between talk vectors.
    - Created a function to retrieve the top N most similar talks for any given TED talk in the database.

## 📊 Dataset

The project uses `tedx_datase.csv` containing over 4,400 TED talks with the following information:
- `main_speaker`: The person giving the talk.
- `title`: The headline of the talk.
- `details`: A detailed description/abstract of the talk.
- `posted`: Date information.
- `url`: Link to the actual talk on TED.com.

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "Ted Talks Recommendation System with Machine Learning"
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib nltk scikit-learn wordcloud
   ```

3. **Launch the Notebook:**
   Open `main.ipynb` in your Jupyter environment and run the cells to see the recommender in action.

## 📈 Key Insights

- The system effectively identifies talks with similar philosophical, scientific, or technological themes.
- TF-IDF successfully filters out common noise and highlights specific topical keywords that define a talk's unique value.

---
Developed with ❤️ by [Harsh Choudhary](https://github.com/HarshChoudhary2003)
