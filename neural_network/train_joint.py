from autoencoder import Autoencoder
from MLP import MLP
import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import MoviesDataset
from torch.utils.data import DataLoader
import os
from tqdm import trange
import wandb
import datetime


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        save_path = os.path.join(
            os.path.dirname(__file__),
            'models/joint/',
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )

        DATA_PATH = os.path.join(os.path.dirname(__file__), '../data')
        os.chdir(DATA_PATH)
        movies_path = 'raw_data/movies.csv'
        ratings_path = 'raw_data/ratings.csv'
        merged_path = 'data.csv'

        dataset = MoviesDataset(movies_path, ratings_path, merged_path, config.movie_context_len)
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        input_dim = dataset.compute_user_embed(1).numel()
        movie_dim = dataset.movie_embeds[1].numel()

        autoencoder = Autoencoder(input_dim, config.hidden_dim).to(device)
        mlp = MLP(input_dim=config.hidden_dim + movie_dim, hidden_dim=10, output_dim=1).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(autoencoder.parameters()) + list(mlp.parameters()), lr=config.lr)
        best_test_loss = float('inf')

        for epoch in trange(1, config.num_epochs + 1):
            autoencoder.train()
            mlp.train()

            train_total_loss = 0
            train_ae_loss = 0
            train_mlp_loss = 0
            squared_error = 0
            abs_error = 0
            total_samples = 0

            for movie_embed, user_embed, rating, *_ in train_dataloader:
                movie_embed, user_embed, rating = movie_embed.to(device), user_embed.to(device), rating.to(device).view(-1, 1)

                latent = autoencoder.encode(user_embed)
                reconstruction = autoencoder.decode(latent)

                loss_ae = criterion(reconstruction, user_embed)
                predicted_rating = mlp(torch.hstack((latent, movie_embed)))
                loss_mlp = criterion(predicted_rating, rating)
                loss = loss_ae + loss_mlp

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_total_loss += loss.item()
                train_ae_loss += loss_ae.item()
                train_mlp_loss += loss_mlp.item()
                abs_error += torch.abs(predicted_rating - rating).sum().item()
                squared_error += torch.sum((predicted_rating - rating) ** 2).item()
                total_samples += rating.size(0)

            wandb.log({
                "train/loss_total": train_total_loss / len(train_dataloader),
                "train/loss_autoencoder": train_ae_loss / len(train_dataloader),
                "train/loss_mlp": train_mlp_loss / len(train_dataloader),
                "train/mae": abs_error / total_samples,
                "train/rmse": (squared_error / total_samples) ** 0.5,
                "epoch": epoch
            })

            # --- Evaluation ---
            autoencoder.eval()
            mlp.eval()
            test_total_loss = 0
            test_ae_loss = 0
            test_mlp_loss = 0
            test_abs_error = 0
            test_squared_error = 0
            test_samples = 0

            with torch.no_grad():
                for movie_embed, user_embed, rating, *_ in test_dataloader:
                    movie_embed, user_embed, rating = movie_embed.to(device), user_embed.to(device), rating.to(device).view(-1, 1)

                    latent = autoencoder.encode(user_embed)
                    reconstruction = autoencoder.decode(latent)

                    loss_ae = criterion(reconstruction, user_embed)
                    predicted_rating = mlp(torch.hstack((latent, movie_embed)))
                    loss_mlp = criterion(predicted_rating, rating)
                    loss = loss_ae + loss_mlp

                    test_total_loss += loss.item()
                    test_ae_loss += loss_ae.item()
                    test_mlp_loss += loss_mlp.item()
                    test_abs_error += torch.abs(predicted_rating - rating).sum().item()
                    test_squared_error += torch.sum((predicted_rating - rating) ** 2).item()
                    test_samples += rating.size(0)

            wandb.log({
                "test/loss_total": test_total_loss / len(test_dataloader),
                "test/loss_autoencoder": test_ae_loss / len(test_dataloader),
                "test/loss_mlp": test_mlp_loss / len(test_dataloader),
                "test/mae": test_abs_error / test_samples,
                "test/rmse": (test_squared_error / test_samples) ** 0.5,
                "epoch": epoch
            })

            if test_total_loss < best_test_loss:
                best_test_loss = test_total_loss
                this_save_path = os.path.join(save_path, f'epoch_{epoch}')
                os.makedirs(this_save_path, exist_ok=True)
                torch.save(autoencoder.state_dict(), os.path.join(this_save_path, 'autoencoder.pth'))
                torch.save(mlp.state_dict(), os.path.join(this_save_path, 'mlp.pth'))
                wandb.save(os.path.join(this_save_path, 'autoencoder.pth'))
                wandb.save(os.path.join(this_save_path, 'mlp.pth'))
                print(f"Epoch {epoch}: model saved at {this_save_path}")

if __name__ == "__main__":

    
    config = {
        "batch_size": 256,
        "hidden_dim": 128,
        "lr": 1e-3,
        "num_epochs": 200,
        "movie_context_len": 19
    }
    train(config)