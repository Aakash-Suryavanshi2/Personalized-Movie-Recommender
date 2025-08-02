import torch
import os
import json
import argparse
import itertools
import matplotlib.pyplot as plt
from autoencoder import Autoencoder
from MLP import MLP
from data_utils import MoviesDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from visualize import visualize_latent_space

def load_wandb_config(wandb_path):
    metadata_path = os.path.join(wandb_path, 'files', 'wandb-metadata.json')
    return {
        'movie_context_len': 19,
        'num_epochs': 196,
        'batch_size': 64,
        'hidden_dim': 128,
        'run_path': wandb_path,
    }

def plot_ratings_comparison(true_ratings, pred_ratings):
    true_ratings = true_ratings.cpu().numpy().flatten()
    pred_ratings = pred_ratings.detach().cpu().numpy().flatten()

    # Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(true_ratings, pred_ratings, alpha=0.5, c='blue', label='Predicted vs True')
    plt.plot([0, 5], [0, 5], 'r--', label='Perfect prediction')
    plt.xlabel("True Ratings")
    plt.ylabel("Predicted Ratings")
    plt.title("Predicted vs True Ratings")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("scatter_pred_vs_true.png")
    plt.show()

    # Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(true_ratings, bins=20, alpha=0.5, label='True Ratings')
    plt.hist(pred_ratings, bins=20, alpha=0.5, label='Predicted Ratings')
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of True vs Predicted Ratings")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hist_ratings.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Inference for joint movie recommendation model.')
    parser.add_argument('--wandb_path', type=str, required=True, help='Path to the wandb run directory.')
    args = parser.parse_args()

    config = load_wandb_config(args.wandb_path)

    DATA_PATH = os.path.join(os.path.dirname(__file__), '../data')
    os.chdir(DATA_PATH)
    movies_path = 'raw_data/movies.csv'
    ratings_path = 'raw_data/ratings.csv'
    merged_path = 'data.csv'

    dataset = MoviesDataset(movies_path, ratings_path, merged_path, config['movie_context_len'])
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model_path = os.path.join(config['run_path'], f'files/epoch_{config["num_epochs"]}')
    autoencoder_path = os.path.join(model_path, 'autoencoder.pth')
    mlp_path = os.path.join(model_path, 'mlp.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = dataset.compute_user_embed(1).numel()
    movie_dim = dataset.movie_embeds[1].numel()

    autoencoder = Autoencoder(input_dim, config['hidden_dim']).to(device)
    mlp = MLP(input_dim=config['hidden_dim'] + movie_dim, hidden_dim=10, output_dim=1).to(device)

    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
    mlp.load_state_dict(torch.load(mlp_path, map_location=device))

    autoencoder.eval()
    mlp.eval()

    out = torch.empty((0, config['hidden_dim'])).to(device)
    user_ids = torch.empty((0, 1), dtype=torch.int)
    movie_ids = torch.empty((0, 1), dtype=torch.int)
    ratings = torch.empty((0, 1))
    pred_ratings_all = torch.empty((0, 1))

    for movie_embed, user_embed, rating, movie_id, user_id in tqdm(dataloader):
        movie_embed = movie_embed.to(device)
        user_embed = user_embed.to(device)
        rating = rating.to(device)

        latent = autoencoder.encode(user_embed)
        pred_ratings = mlp(torch.hstack((latent, movie_embed)))

        out = torch.vstack((out, latent))
        ratings = torch.vstack((ratings, rating.view(-1, 1)))
        pred_ratings_all = torch.vstack((pred_ratings_all, pred_ratings))

        user_ids = torch.vstack((user_ids, user_id.view(-1, 1)))
        movie_ids = torch.vstack((movie_ids, movie_id.view(-1, 1)))

        
    # Print and visualize predictions
    print(f"Predicted vs True Ratings Sample:\n{torch.hstack((pred_ratings, rating.view(-1,1)))}")
    plot_first_n_samples = 600
    plot_ratings_comparison(ratings[:plot_first_n_samples], pred_ratings_all[:plot_first_n_samples])

    visualize_latent_space(out[:plot_first_n_samples],
                            movie_ids[:plot_first_n_samples],
                            user_ids[:plot_first_n_samples],
                            ratings[:plot_first_n_samples],
                            color_by='user_ids', )

    import scipy.stats as stats
    from sklearn.metrics import r2_score
    ratings = ratings.flatten().detach().numpy()
    pred_ratings = pred_ratings_all.flatten().detach().numpy()
    corr, p_value = stats.pearsonr(ratings, pred_ratings)

    print("Evaluation Metrics:")
    r2 = r2_score(ratings, pred_ratings)
    print(f"  R² Score: {r2:.4f}")
    print(f"  Pearson Correlation: {corr:.4f} (p={p_value:.4e})")

if __name__ == "__main__":
    main()
