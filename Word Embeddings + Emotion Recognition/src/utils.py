import re
import string
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
from sklearn.metrics import confusion_matrix

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def setup_logging(log_file='project.log'):
    """Sets up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('NLP_Project')

def clean_text(text, keep_hyphens=True):
    """
    Performs preprocessing:
    - lowercase conversion
    - remove special characters while keeping hyphens (optional)
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    
    if keep_hyphens:
        # Keep letters, numbers, hyphens, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s-]', '', text)
    else:
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_corpus(text):
    """
    Sentence and word tokenization.
    """
    sentences = sent_tokenize(text)
    tokenized_sentences = [word_tokenize(clean_text(sent)) for sent in sentences]
    # Filter out empty sentences/tokens
    tokenized_sentences = [s for s in tokenized_sentences if len(s) > 0]
    return tokenized_sentences

def save_plot(fig, filename, output_dir='outputs/plots'):
    """Saves a matplotlib figure to the specified directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot to {path}")

def get_stop_words(keep_negations=True):
    """
    Returns a set of stop words, optionally keeping 'not' and 'no'.
    """
    stop_words = set(stopwords.words('english'))
    if keep_negations:
        stop_words.discard('not')
        stop_words.discard('no')
        stop_words.discard('never')
        stop_words.discard('neither')
        stop_words.discard('nor')
    return stop_words

def plot_confusion_matrix_custom(y_true, y_pred, labels, filename, title="Confusion Matrix"):
    """Generates and saves a confusion matrix heatmap."""
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    save_plot(plt.gcf(), filename)

def plot_f1_scores(report, filename):
    """Generates a bar chart of F1 scores per class."""
    classes = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    f1_scores = [report[k]['f1-score'] for k in classes]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=classes, y=f1_scores, palette='viridis')
    plt.xticks(rotation=90)
    plt.title("F1 Scores per Class")
    plt.ylabel("F1 Score")
    save_plot(plt.gcf(), filename)

def plot_error_analysis_length(texts, y_true, y_pred, filename):
    """Analyzes error distribution by comment length."""
    lengths = [len(str(t).split()) for t in texts]
    is_correct = [int(t == p) for t, p in zip(y_true, y_pred)]
    
    df = pd.DataFrame({'Length': lengths, 'Correct': is_correct})
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Correct', y='Length', data=df)
    plt.title("Impact of Comment Length on Classification Accuracy")
    save_plot(plt.gcf(), filename)

def plot_most_confused(y_true, y_pred, labels, filename, top_n=10):
    """Visualizes most frequently confused emotion pairs."""
    cm = confusion_matrix(y_true, y_pred)
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and i < len(labels) and j < len(labels):
                confusions.append((labels[i], labels[j], cm[i, j]))
    
    confusions.sort(key=lambda x: x[2], reverse=True)
    top_confusions = confusions[:top_n]
    
    pairs = [f"{c[0]} -> {c[1]}" for c in top_confusions]
    counts = [c[2] for c in top_confusions]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=pairs, palette='magma')
    plt.title(f"Top {top_n} Most Confused Emotion Pairs")
    save_plot(plt.gcf(), filename)
