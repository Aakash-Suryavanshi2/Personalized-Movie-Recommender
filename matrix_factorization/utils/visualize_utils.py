import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_pca(V, movie_ids, movies_df, savefig=False):
    pca = PCA(n_components=2)
    V_2D = pca.fit_transform(V)

    movies_df = movies_df[movies_df['movieId'].isin(movie_ids)].set_index('movieId').loc[movie_ids].reset_index()
    movies_df['pca_x'] = V_2D[:, 0]
    movies_df['pca_y'] = V_2D[:, 1]

    plt.figure(figsize=(12, 8))
    plt.scatter(movies_df['pca_x'], movies_df['pca_y'], s=10, alpha=0.5)
    plt.title("Movie Latent Space (PCA Projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig("../visuals/latent_space/movie_latent_space_pca.png")
    plt.show()

def plot_tsne(V, movie_ids, movies_df, savefig=False):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    V_tsne = tsne.fit_transform(V)

    movies_df = movies_df[movies_df['movieId'].isin(movie_ids)].set_index('movieId').loc[movie_ids].reset_index()
    movies_df['tsne_x'] = V_tsne[:, 0]
    movies_df['tsne_y'] = V_tsne[:, 1]

    plt.figure(figsize=(12, 8))
    plt.scatter(movies_df['tsne_x'], movies_df['tsne_y'], s=10, alpha=0.5)
    plt.title("Movie Latent Space (t-SNE Projection)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig("../visuals/latent_space/movie_latent_space_tsne.png")
    plt.show()

def plot_pca_genre(V, movie_ids, movies_df, genre, savefig=False):
    pca = PCA(n_components=2)
    V_2D = pca.fit_transform(V)

    movies_df = movies_df[movies_df['movieId'].isin(movie_ids)].set_index('movieId').loc[movie_ids].reset_index()
    movies_df['pca_x'] = V_2D[:, 0]
    movies_df['pca_y'] = V_2D[:, 1]
    movies_df['in_genre'] = movies_df['genres'].str.contains(genre, case=False, na=False)

    plt.figure(figsize=(12, 8))
    plt.scatter(movies_df['pca_x'], movies_df['pca_y'], s=10, alpha=0.2, color='gray', label='All Movies')
    plt.scatter(movies_df[movies_df['in_genre']]['pca_x'], movies_df[movies_df['in_genre']]['pca_y'], color='red', s=20, label=f'"{genre}" Movies')
    plt.title(f'{genre} Movies in Latent Space (PCA)')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig(f"../visuals/latent_space/pca_{genre.lower()}_highlighted.png")
    plt.show()

def plot_tsne_genre(V, movie_ids, movies_df, genre, savefig=False):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    V_tsne = tsne.fit_transform(V)

    movies_df = movies_df[movies_df['movieId'].isin(movie_ids)].set_index('movieId').loc[movie_ids].reset_index()
    movies_df['tsne_x'] = V_tsne[:, 0]
    movies_df['tsne_y'] = V_tsne[:, 1]
    movies_df['in_genre'] = movies_df['genres'].str.contains(genre, case=False, na=False)

    plt.figure(figsize=(12, 8))
    plt.scatter(movies_df['tsne_x'], movies_df['tsne_y'], s=10, alpha=0.2, color='gray', label='All Movies')
    plt.scatter(movies_df[movies_df['in_genre']]['tsne_x'], movies_df[movies_df['in_genre']]['tsne_y'], color='red', s=20, label=f'"{genre}" Movies')
    plt.title(f'{genre} Movies in Latent Space (t-SNE)')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig(f"../visuals/latent_space/tsne_{genre.lower()}_highlighted.png")
    plt.show()

def plot_pca_ratings(V, movie_ids, movies_df, ratings_df, threshold=4.5, savefig=False):
    pca = PCA(n_components=2)
    V_2D = pca.fit_transform(V)

    movies_df = movies_df[movies_df['movieId'].isin(movie_ids)].set_index('movieId').loc[movie_ids].reset_index()
    movies_df['pca_x'] = V_2D[:, 0]
    movies_df['pca_y'] = V_2D[:, 1]

    avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
    movies_df['avg_rating'] = movies_df['movieId'].map(avg_ratings)
    movies_df['highly_rated'] = movies_df['avg_rating'] >= threshold

    plt.figure(figsize=(12, 8))
    plt.scatter(movies_df['pca_x'], movies_df['pca_y'], s=10, alpha=0.2, color='gray', label='All Movies')
    plt.scatter(movies_df[movies_df['highly_rated']]['pca_x'], movies_df[movies_df['highly_rated']]['pca_y'], s=30, color='green', label=f'Avg Rating ≥ {threshold}')
    plt.title(f'Movies with High Average Rating (PCA)')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig(f"../visuals/latent_space/pca_high_rating_{threshold}.png")
    plt.show()

def plot_tsne_ratings(V, movie_ids, movies_df, ratings_df, threshold=4.5, savefig=False):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    V_tsne = tsne.fit_transform(V)

    movies_df = movies_df[movies_df['movieId'].isin(movie_ids)].set_index('movieId').loc[movie_ids].reset_index()
    movies_df['tsne_x'] = V_tsne[:, 0]
    movies_df['tsne_y'] = V_tsne[:, 1]

    avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
    movies_df['avg_rating'] = movies_df['movieId'].map(avg_ratings)
    movies_df['highly_rated'] = movies_df['avg_rating'] >= threshold

    plt.figure(figsize=(12, 8))
    plt.scatter(movies_df['tsne_x'], movies_df['tsne_y'], s=10, alpha=0.2, color='gray', label='All Movies')
    plt.scatter(movies_df[movies_df['highly_rated']]['tsne_x'], movies_df[movies_df['highly_rated']]['tsne_y'], s=30, color='green', label=f'Avg Rating ≥ {threshold}')
    plt.title(f'Movies with High Average Rating (t-SNE)')
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig(f"../visuals/latent_space/tsne_high_rating_{threshold}.png")
    plt.show()