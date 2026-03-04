"""
nltk_pipeline.py
Complete NLTK preprocessing pipeline:
- tokenization (word + sentence)
- stopword removal
- custom stopword handling (preserve negations)
- stemming (Porter, Snowball) and lemmatization (WordNet)
- POS tagging & extraction
- export processed CSV
"""

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag
from typing import List
import time
import json
from preprocessing_library import clean_text

# Ensure NLTK data was downloaded (see README)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')

PORTER = PorterStemmer()
SNOW = SnowballStemmer('english')
LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))
# ensure negations are kept
NEGATIONS = {'no', 'not', 'never'}
STOPWORDS = STOPWORDS - NEGATIONS

def pos_tag_to_wordnet_pos(treebank_tag):
    """Convert TreeBank POS to WordNet POS for lemmatization"""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default

def nltk_tokenize(text: str) -> List[str]:
    return word_tokenize(text)

def nltk_sentence_tokenize(text: str) -> List[str]:
    return sent_tokenize(text)

def remove_stopwords(tokens: List[str], custom_stopwords: set = None) -> List[str]:
    if custom_stopwords is None:
        custom_stopwords = set()
    return [t for t in tokens if t.lower() not in STOPWORDS and t.lower() not in custom_stopwords]

def apply_stemmers(tokens: List[str]) -> dict:
    return {
        'porter': [PORTER.stem(t) for t in tokens],
        'snowball': [SNOW.stem(t) for t in tokens]
    }

def apply_lemmatization(tokens: List[str], pos_tags=None) -> List[str]:
    if pos_tags is None:
        pos_tags = pos_tag(tokens)
    lemmas = []
    for token, tag in pos_tags:
        wn_tag = pos_tag_to_wordnet_pos(tag)
        lemmas.append(LEMMATIZER.lemmatize(token, wn_tag))
    return lemmas

def extract_nouns_adjectives(tokens: List[str]) -> List[str]:
    tags = pos_tag(tokens)
    return [tok for tok, tg in tags if tg.startswith('N') or tg.startswith('J')]

# -------------- Complete pipeline function --------------
def nltk_pipeline_dataframe(df: pd.DataFrame,
                            text_col: str = 'text',
                            output_csv: str = 'tweets_nltk_processed.csv',
                            custom_stopwords: set = None):
    start = time.time()
    records = []
    for idx, row in df.iterrows():
        raw = row[text_col]
        cleaned = clean_text(raw,
                             lowercase_flag=True,
                             remove_urls_flag=True,
                             remove_mentions_flag=True,
                             remove_hashtags_flag=False,
                             keep_hashtag_text=True,
                             remove_numbers_flag=False,
                             remove_emoji_flag=False,
                             emoji_to_text_flag=False,
                             expand_contractions_flag=True)

        # tokenization
        words = nltk_tokenize(cleaned)
        sents = nltk_sentence_tokenize(cleaned)

        # filtered tokens
        tokens_filtered = remove_stopwords(words, custom_stopwords=custom_stopwords)

        # stem & lemma
        stems = apply_stemmers(tokens_filtered)
        pos_tags = pos_tag(tokens_filtered)
        lemmas = apply_lemmatization(tokens_filtered, pos_tags=pos_tags)

        # POS extraction
        nouns_adjs = extract_nouns_adjectives(tokens_filtered)

        records.append({
            'original': raw,
            'cleaned': cleaned,
            'tokens': words,
            'tokens_filtered': tokens_filtered,
            'lemmas': lemmas,
            'stems_porter': stems['porter'],
            'stems_snowball': stems['snowball'],
            'pos_tags': pos_tags,
            'nouns_adjs': nouns_adjs
        })

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)
    elapsed = time.time() - start
    print(f"NLTK pipeline processed {len(df)} rows in {elapsed:.2f} seconds. Output saved to {output_csv}")
    return df_out

# -------------- If run as script --------------
if __name__ == '__main__':
    import sys
    path = 'Tweets.csv' if len(sys.argv) < 2 else sys.argv[1]
    df = pd.read_csv(path)
    # use only first 10000 rows for assignment
    df = df.head(10000)
    processed = nltk_pipeline_dataframe(df, text_col='text', output_csv='tweets_nltk_processed.csv')
