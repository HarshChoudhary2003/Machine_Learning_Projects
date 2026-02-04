# 🍽️ Restaurant Review Sentiment Analysis (NLP)

A machine learning project that analyzes restaurant reviews to determine if they are positive or negative. This project uses Natural Language Processing (NLP) techniques like text cleaning, stemming, and the Bag of Words model to build an accurate classifier.

---

## ✨ Features

- **🧹 Data Cleaning**: Removes special characters and numbers from reviews.
- **🔡 Text Normalization**: Converts all text to lowercase for consistency.
- **✂️ Stemming**: Uses `PorterStemmer` to reduce words to their root forms (e.g., "loved" -> "love").
- **🚫 Stopword Removal**: Filters out common English words that don't add semantic value.
- **👜 Bag of Words Model**: Transforms text into numerical feature vectors using `CountVectorizer`.
- **🧠 Advanced Classifiers**: Implements and compares multiple models including:
  - Random Forest Classifier (Entropy & Gini)
  - Multinomial Naive Bayes
  - Support Vector Machine (SVM)

---

## 🛠️ Technology Stack

- **Python 3.x**
- **Libraries**:
  - `pandas` & `numpy`: Data manipulation.
  - `nltk`: Natural Language Toolkit for text processing.
  - `scikit-learn`: For feature extraction and machine learning models.
  - `re`: Regular expressions for text cleaning.

---

## 📊 Dataset

The project uses the `Restaurant_Reviews.tsv` file, which contains:
- **Review**: The text of the customer review.
- **Liked**: A binary label (1 for positive, 0 for negative).

---

## 🚀 Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HarshChoudhary2003/Machine_Learning_Projects.git
   cd "NLP analysis of Restaurant reviews"
   ```

2. **Install Dependencies**:
   ```bash
   pip install pandas numpy nltk scikit-learn
   ```

3. **Run the Analysis**:
   Open the `main.ipynb` notebook and execute the cells to see the data preprocessing and model training.

---

## 📈 Results

The project evaluates model performance using:
- **Confusion Matrix**
- **Accuracy Score**
- **Classification Report** (Precision, Recall, F1-Score)

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
