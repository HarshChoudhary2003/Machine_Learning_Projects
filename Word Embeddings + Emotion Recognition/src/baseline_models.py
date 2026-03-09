import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, hamming_loss
from utils import setup_logging, save_plot

logger = setup_logging()

class BaselineModels:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced')
        }
        self.results = {}

    def prepare_data(self, X_train, X_test):
        """Vectorizes text data using TF-IDF."""
        logger.info("Vectorizing data with TF-IDF...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        return X_train_tfidf, X_test_tfidf

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Trains and evaluates baseline models."""
        X_tr, X_te = self.prepare_data(X_train, X_test)
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_tr, y_train)
            preds = model.predict(X_te)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            h_loss = hamming_loss(y_test, preds)
            
            self.results[name] = {
                'accuracy': acc,
                'f1_macro': f1,
                'hamming_loss': h_loss,
                'report': classification_report(y_test, preds, output_dict=True)
            }
            
            logger.info(f"{name} - Acc: {acc:.4f}, F1 Macro: {f1:.4f}")
            
            # Confusion Matrix
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(y_test, preds)
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
            plt.title(f"Confusion Matrix - {name}")
            save_plot(plt.gcf(), f"cm_{name.lower()}.png")

        return self.results

    def compare_performances(self):
        """Generates a comparison chart."""
        names = list(self.results.keys())
        accs = [self.results[n]['accuracy'] for n in names]
        f1s = [self.results[n]['f1_macro'] for n in names]
        
        df = pd.DataFrame({'Model': names, 'Accuracy': accs, 'F1-Macro': f1s})
        plt.figure(figsize=(10, 6))
        df.plot(x='Model', kind='bar', subplots=False, figsize=(10, 6))
        plt.title("Baseline Model Comparison")
        save_plot(plt.gcf(), "baseline_comparison.png")
        return df

if __name__ == "__main__":
    # Test with dummy data if needed
    X = ["I am so happy", "I hate this", "What a surprise", "This is okay"] * 10
    y = [1, 2, 3, 0] * 10
    baselines = BaselineModels()
    baselines.train_and_evaluate(X, y, X, y)
    baselines.compare_performances()
