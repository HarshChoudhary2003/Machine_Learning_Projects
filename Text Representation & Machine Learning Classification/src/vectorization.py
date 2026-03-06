import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():

    X_train = pickle.load(open("data/processed/X_train.pkl", "rb"))
    X_test = pickle.load(open("data/processed/X_test.pkl", "rb"))

    return X_train, X_test


def bow_vectorizer():

    vectorizer = CountVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.9,
        binary=False,
        ngram_range=(1,1)
    )

    return vectorizer


def tfidf_vectorizer():

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2),
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )

    return vectorizer


def char_vectorizer():

    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(2,4),
        max_features=5000
    )

    return vectorizer


def transform_data(vectorizer, X_train, X_test):

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec