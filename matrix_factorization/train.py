import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model import MatrixFactorizationModel
from tune_hyperparameters import tune_num_features, tune_alpha, tune_beta, tune_epochs
from utils.visualize_utils import plot_pca, plot_tsne, plot_pca_genre, plot_tsne_genre, plot_pca_ratings, plot_tsne_ratings
from utils.matrix_utils import split_matrix, display_top_k_recommendations
import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparameters', action='store_true', help="Flag to tune hyperparameters")
    args = parser.parse_args()
    tune_hyperparameters = args.hyperparameters

    DATA_PATH = os.path.join(__file__,'../../data')
        
    os.chdir(DATA_PATH)
    ratings_df = pd.read_csv("data.csv")
    movies_df = pd.read_csv("movies.csv")
   
    user_item_matrix = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    R = user_item_matrix.values
    num_users, num_items = R.shape
    movie_ids = list(user_item_matrix.columns)
    train_R, test_R = split_matrix(R)

    if tune_hyperparameters:
        tune_num_features(R, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
        tune_alpha(R, [0.00025, 0.0005, 0.00075, 0.001, 0.002, 0.003, 0.004, 0.005])
        tune_beta(R, [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
        tune_epochs(R, 200)

    alpha = 0.002
    beta = 0.04
    num_features = 100
    epochs = 60
    K = 5

    model = MatrixFactorizationModel(num_users, num_items, num_features, alpha, beta)
    model.train(train_R, epochs=epochs)
    predicted_ratings = model.predict()

    test_mask = test_R > 0
    actual = test_R[test_mask]
    predicted = predicted_ratings[test_mask]
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    print(f"\nFinal Evaluation:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    for user_index in [0, 4, 9]:
        display_top_k_recommendations(user_index, predicted_ratings, train_R, movie_ids, movies_df,user_item_matrix)

    plot_pca(model.V, movie_ids, movies_df)
    plot_tsne(model.V, movie_ids, movies_df)
    plot_pca_genre(model.V, movie_ids, movies_df, "Crime")
    plot_tsne_genre(model.V, movie_ids, movies_df, "Crime")
    plot_pca_ratings(model.V, movie_ids, movies_df, ratings_df, 4.5)
    plot_tsne_ratings(model.V, movie_ids, movies_df, ratings_df, 4.5)