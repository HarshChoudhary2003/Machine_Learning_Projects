"""
spacy_pipeline.py
Complete spaCy preprocessing pipeline:
- load nlp model
- tokenization/lemmatization/POS/DEP in a single pass
- NER extraction + counts
- custom entity ruler example
- export processed CSV and entities JSON
"""

import spacy
from spacy.pipeline import EntityRuler
import pandas as pd
import time
import json
from typing import List
from preprocessing_library import clean_text

# Load model (ensure downloaded with: python -m spacy download en_core_web_sm)
nlp = spacy.load('en_core_web_sm')

def add_custom_entities(nlp, patterns):
    ruler = EntityRuler(nlp, overwrite_ents=True)
    ruler.add_patterns(patterns)
    nlp.add_pipe(ruler, before='ner')

def spacy_process_text(text: str, remove_stopwords: bool = True, custom_stopwords: set = None):
    # First apply the same cleaning rules (we rely on preprocessing_library for noisy text)
    cleaned = clean_text(text,
                         lowercase_flag=False,  # keep case for NER sometimes; optional
                         remove_urls_flag=True,
                         remove_mentions_flag=True,
                         remove_hashtags_flag=False,
                         keep_hashtag_text=True,
                         remove_numbers_flag=False,
                         remove_emoji_flag=False,
                         emoji_to_text_flag=True,  # convert emojis to words for spaCy to see
                         expand_contractions_flag=True)

    doc = nlp(cleaned)

    tokens = [tok.text for tok in doc]
    lemmas = [tok.lemma_ for tok in doc]
    pos = [tok.pos_ for tok in doc]
    pos_tags = [tok.tag_ for tok in doc]
    deps = [tok.dep_ for tok in doc]
    is_stop = [tok.is_stop for tok in doc]

    # filter tokens optionally
    if remove_stopwords:
        if custom_stopwords is None:
            filtered = [tok.text for tok in doc if not tok.is_stop]
        else:
            filtered = [tok.text for tok in doc if (not tok.is_stop and tok.text.lower() not in custom_stopwords)]
    else:
        filtered = [tok.text for tok in doc]

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    ents_by_type = {}
    for e, lab in entities:
        ents_by_type.setdefault(lab, 0)
        ents_by_type[lab] += 1

    return {
        'original': text,
        'cleaned': cleaned,
        'tokens': tokens,
        'lemmas': lemmas,
        'pos': pos,
        'pos_tags': pos_tags,
        'deps': deps,
        'filtered_tokens': filtered,
        'entities': entities,
        'entity_counts': ents_by_type
    }

def spacy_pipeline_dataframe(df: pd.DataFrame,
                             text_col: str = 'text',
                             output_csv: str = 'tweets_spacy_processed.csv',
                             entities_json: str = 'tweets_spacy_entities.json',
                             custom_entity_patterns: List[dict] = None):
    # optionally add custom entities (patterns list of dicts for EntityRuler)
    if custom_entity_patterns:
        add_custom_entities(nlp, custom_entity_patterns)

    start = time.time()
    records = []
    entities_total = []
    for idx, row in df.iterrows():
        rec = spacy_process_text(row[text_col], remove_stopwords=True)
        records.append(rec)
        # collect entities for JSON
        entities_total.append({'original': row[text_col], 'entities': rec['entities']})

    df_out = pd.DataFrame(records)
    df_out.to_csv(output_csv, index=False)

    # dump entities to JSON
    with open(entities_json, 'w', encoding='utf-8') as fh:
        json.dump(entities_total, fh, ensure_ascii=False, indent=2)

    elapsed = time.time() - start
    print(f"spaCy pipeline processed {len(df)} rows in {elapsed:.2f} seconds. Output saved to {output_csv} and {entities_json}")
    return df_out

# --------------- If run as script -----------------
if __name__ == '__main__':
    import sys
    path = 'Tweets.csv' if len(sys.argv) < 2 else sys.argv[1]
    df = pd.read_csv(path)
    df = df.head(10000)
    # example custom entity:
    patterns = [{"label": "AIRLINE", "pattern": "United Airlines"},
                {"label": "AIRLINE", "pattern": "Delta"}]
    processed = spacy_pipeline_dataframe(df, text_col='text',
                                         output_csv='tweets_spacy_processed.csv',
                                         entities_json='tweets_spacy_entities.json',
                                         custom_entity_patterns=patterns)
