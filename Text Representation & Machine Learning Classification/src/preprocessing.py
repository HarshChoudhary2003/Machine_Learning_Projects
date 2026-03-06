import pandas as pd
import re
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    return text


def preprocess_text(text):

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)


def load_dataset():

    df = pd.read_csv("data/raw/IMDB Dataset.csv")

    return df


def process_dataset(df):

    df["clean_review"] = df["review"].apply(clean_text)
    df["processed_review"] = df["clean_review"].apply(preprocess_text)

    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    return df


def split_data(df):

    X = df["processed_review"]
    y = df["sentiment"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_data(X_train, X_val, X_test, y_train, y_val, y_test):

    pickle.dump(X_train, open("data/processed/X_train.pkl", "wb"))
    pickle.dump(X_val, open("data/processed/X_val.pkl", "wb"))
    pickle.dump(X_test, open("data/processed/X_test.pkl", "wb"))

    pickle.dump(y_train, open("data/processed/y_train.pkl", "wb"))
    pickle.dump(y_val, open("data/processed/y_val.pkl", "wb"))
    pickle.dump(y_test, open("data/processed/y_test.pkl", "wb"))


if __name__ == "__main__":

    df = load_dataset()

    df = process_dataset(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    save_data(X_train, X_val, X_test, y_train, y_val, y_test)

    print("Preprocessing completed")