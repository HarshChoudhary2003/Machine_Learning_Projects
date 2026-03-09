import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from utils import setup_logging, save_plot

logger = setup_logging()

from gensim.models import Word2Vec, FastText
from sklearn.metrics.pairwise import cosine_similarity

logger = setup_logging()

class EmbeddingAnalyzer:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        """Loads a trained Word2Vec or FastText model."""
        if 'fasttext' in model_path.lower():
            self.model = FastText.load(model_path)
        else:
            self.model = Word2Vec.load(model_path)
        return self.model

    def extract_top_words(self, top_n=500):
        """Extracts top words from the vocabulary."""
        words = list(self.model.wv.index_to_key)[:top_n]
        vectors = np.array([self.model.wv[w] for w in words])
        return words, vectors

    def plot_similarity_heatmap(self, words, filename_prefix='similarity_heatmap'):
        """Plots a heatmap of cosine similarities between given words."""
        logger.info(f"Generating similarity heatmap for {len(words)} words...")
        
        # Filter words in vocab
        valid_words = [w for w in words if w in self.model.wv]
        if not valid_words:
            logger.warning("No valid words found for similarity heatmap.")
            return

        vectors = np.array([self.model.wv[w] for w in valid_words])
        sim_matrix = cosine_similarity(vectors)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(sim_matrix, annot=True, xticklabels=valid_words, yticklabels=valid_words, cmap='YlGnBu')
        plt.title(f"Cosine Similarity Heatmap ({filename_prefix})")
        save_plot(plt.gcf(), f"{filename_prefix}.png")

    def plot_tsne(self, words, vectors, n_clusters=10, filename_prefix='w2v_tsne'):
        """
        Performs t-SNE dimensionality reduction and generates 2D visualization.
        Color codes words by semantic clusters using KMeans.
        """
        logger.info(f"Performing t-SNE on {len(words)} words...")
        
        # Apply t-SNE
        perplexity = min(30, len(words) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
        vectors_2d = tsne.fit_transform(vectors)
        
        # Cluster for coloring
        n_clusters_adj = min(n_clusters, len(words))
        kmeans = KMeans(n_clusters=n_clusters_adj, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(vectors)
        
        # Generate 2D Plot
        plt.figure(figsize=(16, 10))
        sns.set_style("whitegrid")
        scatter = plt.scatter(
            vectors_2d[:, 0], 
            vectors_2d[:, 1], 
            c=clusters, 
            cmap='tab20', 
            alpha=0.7, 
            edgecolors='k'
        )
        
        # Annotate a subset of words to avoid clutter
        # We'll label about 15% of the points
        for i, word in enumerate(words):
            if i % 7 == 0:
                plt.annotate(
                    word, 
                    xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                    fontsize=9, 
                    alpha=0.8
                )
                
        plt.title(f"t-SNE Visualization of Embeddings ({filename_prefix})")
        plt.colorbar(scatter, label='Cluster ID')
        
        save_plot(plt.gcf(), f"{filename_prefix}_2d.png")

    def plot_tsne_3d(self, words, vectors, n_clusters=10, filename_prefix='w2v_tsne'):
        """Generates 3D visualization using t-SNE."""
        from mpl_toolkits.mplot3d import Axes3D
        
        logger.info(f"Performing 3D t-SNE on {len(words)} words...")
        perplexity = min(30, len(words) - 1)
        tsne = TSNE(n_components=3, perplexity=perplexity, max_iter=1000, random_state=42)
        vectors_3d = tsne.fit_transform(vectors)
        
        n_clusters_adj = min(n_clusters, len(words))
        kmeans = KMeans(n_clusters=n_clusters_adj, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(vectors)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(
            vectors_3d[:, 0], 
            vectors_3d[:, 1], 
            vectors_3d[:, 2], 
            c=clusters, 
            cmap='tab20', 
            alpha=0.7
        )
        
        ax.set_title(f"3D t-SNE Visualization ({filename_prefix})")
        
        # Because we can't interact, we'll just save one view
        save_plot(fig, f"{filename_prefix}_3d.png")

if __name__ == "__main__":
    analyzer = EmbeddingAnalyzer()
    # Assuming at least one model exists from the previous step
    model_dir = 'models/embeddings'
    if os.path.exists(model_dir) and os.listdir(model_dir):
        model_path = os.path.join(model_dir, os.listdir(model_dir)[0])
        analyzer.load_model(model_path)
        words, vectors = analyzer.extract_top_words(top_n=300)
        analyzer.plot_tsne(words, vectors, filename_prefix='emb_sample')
        analyzer.plot_similarity_heatmap(['king', 'queen', 'man', 'woman', 'economy', 'growth'])
    else:
        logger.warning(f"No models found in {model_dir} to analyze.")
