import os
import logging
from gensim.models import Word2Vec
import multiprocessing
from utils import setup_logging, tokenize_corpus, clean_text

logger = setup_logging()

from gensim.models import Word2Vec, FastText

logger = setup_logging()

class EmbeddingsTrainer:
    def __init__(self, sentences=None):
        self.sentences = sentences
        self.models = {}

    def load_corpus_from_file(self, file_path):
        """Loads and tokenizes a corpus from a text file."""
        logger.info(f"Loading corpus from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.sentences = tokenize_corpus(text)
        logger.info(f"Tokenized {len(self.sentences)} sentences.")
        return self.sentences

    def train_variants(self, output_dir='models/embeddings', model_type='word2vec'):
        """
        Trains multiple embedding models (Word2Vec or FastText) with different hyperparameters.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        vector_sizes = [100]
        windows = [5]
        architectures = [1] 
        epochs_list = [5]
        
        count = 0
        total = len(vector_sizes) * len(windows) * len(architectures) * len(epochs_list)
        logger.info(f"Starting grid training for {total} {model_type} variants...")

        for size in vector_sizes:
            for win in windows:
                for arch in architectures:
                    for ep in epochs_list:
                        arch_name = "skipgram" if arch == 1 else "cbow"
                        model_name = f"{model_type}_{arch_name}_sz{size}_win{win}_ep{ep}"
                        model_path = os.path.join(output_dir, f"{model_name}.model")
                        
                        if os.path.exists(model_path):
                            logger.info(f"Model {model_name} already exists. Skipping.")
                            continue

                        logger.info(f"Training {model_name}...")
                        
                        if model_type == 'word2vec':
                            model = Word2Vec(
                                sentences=self.sentences,
                                vector_size=size,
                                window=win,
                                sg=arch,
                                epochs=ep,
                                min_count=5,
                                workers=multiprocessing.cpu_count()
                            )
                        elif model_type == 'fasttext':
                            model = FastText(
                                sentences=self.sentences,
                                vector_size=size,
                                window=win,
                                sg=arch,
                                epochs=ep,
                                min_count=5,
                                workers=multiprocessing.cpu_count()
                            )
                        
                        model.save(model_path)
                        self.models[model_name] = model
                        count += 1
        
        logger.info(f"Finished training {count} new {model_type} variants.")

    def evaluate_model(self, model_path, test_words=['king', 'queen', 'man', 'woman', 'computer', 'science']):
        """Performs evaluation tasks: most_similar, word similarity, analogy."""
        if 'fasttext' in model_path.lower():
            model = FastText.load(model_path)
        else:
            model = Word2Vec.load(model_path)
            
        logger.info(f"Evaluating {model_path}")

        # Similarity example
        for word in test_words:
            if word in model.wv:
                similar = model.wv.most_similar(word, topn=5)
                logger.info(f"Most similar to '{word}': {similar}")
            else:
                logger.warning(f"Word '{word}' not in vocabulary.")

        # Analogy task: king - man + woman = queen
        if all(w in model.wv for w in ['king', 'man', 'woman']):
            analogy = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
            logger.info(f"Analogy: king - man + woman = {analogy}")

if __name__ == "__main__":
    # Create a dummy corpus if none provided for testing
    dummy_text = """
    The economy is growing slowly according to recent news. 
    Technology stocks are fluctuating as computer science advances. 
    A king rules a kingdom while a man lives his life.
    A woman can be a queen in many historical stories.
    Science and computer technology are essential for modern economy.
    """ * 100 # Repeat to meet min_count requirements
    
    os.makedirs('data/corpus', exist_ok=True)
    with open('data/corpus/news_sample.txt', 'w') as f:
        f.write(dummy_text)
        
    trainer = EmbeddingsTrainer()
    trainer.load_corpus_from_file('data/corpus/news_sample.txt')
    trainer.train_variants(model_type='word2vec')
    trainer.train_variants(model_type='fasttext')
    
    # Evaluate one
    w2v_dir = 'models/embeddings'
    if os.listdir(w2v_dir):
        sample_model = os.path.join(w2v_dir, os.listdir(w2v_dir)[1]) # Try FT
        trainer.evaluate_model(sample_model)

