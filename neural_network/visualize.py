import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_latent_space(latents, movie_ids=None, user_ids=None, ratings=None, color_by='ratings'):
    """
    Visualize the latent space using PCA (2D).

    Parameters:
    - latents: Tensor of shape (N, D) – the latent vectors
    - movie_ids, user_ids, ratings: Optional metadata (N,)
    - color_by: One of ['ratings', 'movie_ids', 'user_ids']
    """
    # Convert to numpy
    latents_np = latents.detach().cpu().numpy()
    
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents_np)

    # Choose coloring
    if color_by == 'ratings' and ratings is not None:
        color = ratings.squeeze().cpu().numpy()
        label = 'Rating'
    elif color_by == 'movie_ids' and movie_ids is not None:
        color = movie_ids.squeeze().cpu().numpy()
        label = 'Movie ID'
    elif color_by == 'user_ids' and user_ids is not None:
        color = user_ids.squeeze().cpu().numpy()
        label = 'User ID'
    else:
        color = None
        label = None

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], c=color, cmap='viridis', alpha=0.7)
    if color is not None:
        plt.colorbar(scatter, label=label)
    plt.title(f'Latent Space PCA (colored by {label})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("/Users/alantian/gatech/cs4641/ML4641-38/neural_network/visualizations/latent_space_pca.png")
    plt.show()
    
