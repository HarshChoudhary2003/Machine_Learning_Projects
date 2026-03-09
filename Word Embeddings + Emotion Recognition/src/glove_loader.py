import os
import numpy as np
import logging
from tqdm import tqdm
from utils import setup_logging

logger = setup_logging()

class GloveLoader:
    def __init__(self, glove_path=None):
        self.glove_path = glove_path or 'embeddings/glove.6B.100d.txt'
        self.embeddings_dict = {}

    def load_embeddings(self):
        """Loads glove embeddings into a dictionary."""
        if not os.path.exists(self.glove_path):
            logger.warning(f"GloVe file not found at {self.glove_path}. Creating a dummy file for testing.")
            self._create_dummy_glove()

        logger.info(f"Loading GloVe embeddings from {self.glove_path}...")
        embeddings_dict = {}
        try:
            with open(self.glove_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Loading GloVe"):
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = vector
            self.embeddings_dict = embeddings_dict
            logger.info(f"Loaded {len(embeddings_dict)} GloVe vectors.")
        except Exception as e:
            logger.error(f"Error loading GloVe: {e}")
        return self.embeddings_dict

    def calculate_coverage(self, vocabulary):
        """Calculates vocabulary coverage on dataset."""
        covered = [w for w in vocabulary if w in self.embeddings_dict]
        oov = [w for w in vocabulary if w not in self.embeddings_dict]
        
        coverage = len(covered) / len(vocabulary) if vocabulary else 0
        logger.info(f"Vocabulary coverage: {coverage:.2%} ({len(covered)}/{len(vocabulary)})")
        return coverage, covered, oov

    def create_embedding_matrix(self, word_index, embedding_dim=100):
        """Creates an embedding matrix for the model's vocabulary."""
        vocab_size = len(word_index) + 1
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        
        oov_count = 0
        for word, i in word_index.items():
            embedding_vector = self.embeddings_dict.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                oov_count += 1
                
        logger.info(f"Created matrix with {oov_count} OOV tokens replaced by zero.")
        return embedding_matrix

    def _create_dummy_glove(self, dim=100):
        """Creates a dummy GloVe file with common words for demonstration."""
        os.makedirs(os.path.dirname(self.glove_path), exist_ok=True)
        common_words = ['the', 'a', 'and', 'not', 'no', 'is', 'happy', 'sad', 'angry', 'love', 'joy', 'surprise', 'fear']
        with open(self.glove_path, 'w', encoding='utf-8') as f:
            for word in common_words:
                vec = np.random.normal(0, 0.1, dim)
                f.write(f"{word} {' '.join(vec.astype(str))}\n")
        logger.info(f"Created dummy GloVe at {self.glove_path}")

if __name__ == "__main__":
    loader = GloveLoader()
    loader.load_embeddings()
    sample_vocab = ['the', 'happy', 'unknown_word_abc']
    loader.calculate_coverage(sample_vocab)
