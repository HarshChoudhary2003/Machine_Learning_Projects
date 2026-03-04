
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from preprocessing_library import clean_text
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Setup
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def preprocess_for_ml(text):
    cleaned = clean_text(text, lowercase_flag=True, remove_urls_flag=True, remove_mentions_flag=True)
    tokens = word_tokenize(cleaned)
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(lemmatized)

def train_and_save():
    print("Loading data...")
    df = pd.read_csv("Tweets.csv")
    
    # Filter and clean
    df = df[['airline_sentiment', 'text']]
    df['processed_text'] = df['text'].apply(preprocess_for_ml)
    
    X = df['processed_text']
    y = df['airline_sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Eval
    preds = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))
    
    # Save
    print("Saving model artifacts...")
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Done!")

if __name__ == "__main__":
    train_and_save()
