"""
generate_visuals.py
Small helper functions to create the minimum requested plots using matplotlib.
"""

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

def plot_tweet_length_hist(df, length_col='tweet_length', title='Tweet length distribution', out='tweet_length.png'):
    plt.figure(figsize=(8,5))
    plt.hist(df[length_col].dropna(), bins=40)
    plt.title(title)
    plt.xlabel('Characters')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(out)
    print(f"Saved {out}")
    plt.close()

def plot_processing_time_bar(summary_df, out='processing_time.png'):
    plt.figure(figsize=(8,5))
    sizes = summary_df['size']
    plt.bar([str(x) + ' NLTK' for x in sizes], summary_df['nltk_time_sec'])
    plt.bar([str(x) + ' spaCy' for x in sizes], summary_df['spacy_time_sec'])
    plt.ylabel('Seconds')
    plt.title('Processing time comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out)
    print(f"Saved {out}")
    plt.close()

def wordcloud_from_texts(texts, out='wordcloud.png'):
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(texts))
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out)
    print(f"Saved {out}")
    plt.close()
    return wc