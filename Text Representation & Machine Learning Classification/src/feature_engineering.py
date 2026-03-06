import numpy as np
import pandas as pd


def extract_features(texts):

    features = []

    for text in texts:

        char_len = len(text)

        words = text.split()

        word_count = len(words)

        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        exclamation = text.count("!")

        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text),1)

        features.append([
            char_len,
            word_count,
            avg_word_len,
            exclamation,
            caps_ratio
        ])

    return pd.DataFrame(features, columns=[
        "char_len",
        "word_count",
        "avg_word_len",
        "exclamation",
        "caps_ratio"
    ])