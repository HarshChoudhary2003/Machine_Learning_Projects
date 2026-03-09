import os
import sys
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src'))

from utils import setup_logging
from word2vec_training import EmbeddingsTrainer
from embedding_analysis import EmbeddingAnalyzer
from glove_loader import GloveLoader
from emotion_classifier import EmotionClassifier
from baseline_models import BaselineModels

logger = setup_logging()

def main():
    logger.info("Starting Full Project Execution...")

    # 1. Embeddings Training (Word2Vec & FastText)
    logger.info("--- Step 1: Embeddings Training ---")
    if not os.path.exists('data/corpus/news_sample.txt'):
        os.makedirs('data/corpus', exist_ok=True)
        with open('data/corpus/news_sample.txt', 'w') as f:
            words = ["economy", "technology", "science", "king", "queen", "man", "woman", "computer", "market", "growth"]
            text = " ".join([np.random.choice(words) for _ in range(5000)])
            f.write(text)
    
    emb_trainer = EmbeddingsTrainer()
    emb_trainer.load_corpus_from_file('data/corpus/news_sample.txt')
    emb_trainer.train_variants(model_type='word2vec')
    emb_trainer.train_variants(model_type='fasttext')
    
    # 2. Embedding Analysis
    logger.info("--- Step 2: Embedding Analysis ---")
    emb_files = os.listdir('models/embeddings')
    if emb_files:
        analyzer = EmbeddingAnalyzer()
        # Analyze first model
        model_path = os.path.join('models/embeddings', emb_files[0])
        analyzer.load_model(model_path)
        words, vectors = analyzer.extract_top_words(top_n=100)
        analyzer.plot_tsne(words, vectors, filename_prefix=f'run_{emb_files[0]}_tsne')
        analyzer.plot_similarity_heatmap(['king', 'queen', 'man', 'woman', 'economy', 'growth'], filename_prefix=f'run_{emb_files[0]}_sim')

    # 3. Emotion Classification
    logger.info("--- Step 3: Emotion Classification ---")
    if os.path.exists('go_emotions_dataset.csv'):
        df = pd.read_csv('go_emotions_dataset.csv').head(5000) # Use subset for speed
        emotion_cols = df.columns[3:]
        texts = df['text'].astype(str).values
        labels = df[emotion_cols].values
        
        classifier = EmotionClassifier(max_words=5000, max_len=30)
        X_train, X_val, X_test, y_train, y_val, y_test = classifier.prepare_data(texts, labels, multi_label=True)
        
        # Load GloVe (Dummy if missing)
        loader = GloveLoader()
        loader.load_embeddings()
        classifier.embedding_matrix = loader.create_embedding_matrix(classifier.tokenizer.word_index)
        
        # Train BiLSTM with Attention
        logger.info("Training BiLSTM with Attention...")
        model = classifier.build_bilstm_model(len(emotion_cols), multi_label=True, use_attention=True)
        history = classifier.train_model(model, X_train, y_train, X_val, y_val, epochs=5, multi_label=True)
        classifier.plot_history(history, model_name='BiLSTM_Attention_Run')
        classifier.evaluate(model, X_test, y_test, label_names=emotion_cols, multi_label=True)
        
        # Single Text Prediction Demo
        logger.info("--- Step 4: Prediction Demo ---")
        sample_texts = [
            "I am so happy and excited about this new project!",
            "I feel very sad and disappointed with the results.",
            "This is so annoying and frustrating.",
            "Wow, I didn't expect that! What a surprise!"
        ]
        for txt in sample_texts:
            pred = classifier.predict_single(model, txt, emotion_cols, multi_label=True)
            logger.info(f"Text: '{txt}' -> Predicted Emotions: {pred}")

        # Baseline
        logger.info("--- Step 5: Baseline Comparison ---")
        # Use single label for baselines (top emotion)
        y_tr_s = np.argmax(y_train, axis=1)
        y_te_s = np.argmax(y_test, axis=1)
        
        # For baseline classes we need to use a subset of the original texts that corresponds to the split
        # Since we used random_state=42 in prepare_data, we can replicate the split
        X_train_txt, X_test_txt = train_test_split(texts, test_size=0.3, random_state=42)[:2]
        # Further split X_test_txt to match X_val and X_test
        X_val_txt, X_test_txt = train_test_split(X_test_txt, test_size=0.5, random_state=42)
        
        baselines = BaselineModels()
        baselines.train_and_evaluate(X_train_txt, y_tr_s, X_test_txt, y_te_s)
        baselines.compare_performances()

    logger.info("Full project run complete. Check 'outputs/plots' for results.")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    main()
