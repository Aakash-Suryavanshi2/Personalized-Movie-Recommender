import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import argparse

def plot_neighbor_distance_distribution(knn_model, X_test):
    distances, _ = knn_model.kneighbors(X_test)
    avg_dists = distances.mean(axis=1)

    plt.figure(figsize=(8, 5))
    sns.histplot(avg_dists, bins=40, kde=True, color='steelblue')
    plt.title("Distribution of Average k-NN Distances")
    plt.xlabel("Average Distance to Neighbors")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("visuals/avg_knn_distance_dist.png")
    plt.show()

def plot_ratings_comparison(true_ratings, pred_ratings):
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
    plt.savefig("visuals/scatter_pred_vs_true.png")
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
    plt.savefig("visuals/hist_ratings.png")
    plt.show()


def train(k=100, plot=True):
    df = pd.read_csv('/Users/alantian/gatech/cs4641/ML4641-38/data/data.csv')
    print(f'Knn: {k}')
    # Process genres
    df['genres'] = df['genres'].apply(lambda x: x.split('|'))

    # Train/test split by users
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # One-hot encode genres
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df['genres'])
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)
    df = pd.concat([df, genre_df], axis=1)

    # Compute average genre and rating features per movie
    movie_features = df.groupby('movieId')[mlb.classes_.tolist()].mean().reset_index()
    movie_ratings = df.groupby('movieId')['rating'].mean().reset_index().rename(columns={'rating': 'avg_rating'})
    movie_features = pd.merge(movie_features, movie_ratings, on='movieId')

    # Normalize features and compute cosine-style space
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(movie_features.drop('movieId', axis=1))
    features_cosine = normalize(features_scaled)

    fallback_counter = Counter()
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(features_cosine)


    
    def predict_user_rating_knn(user_id, movie_id):
        row = movie_features[movie_features['movieId'] == movie_id]
        if row.empty:
            fallback_counter['movie_missing'] += 1
            return np.nan

        movie_index = row.index[0]  # Get the index of the movie
        movie_vector = features_cosine[movie_index].reshape(1, -1)

        # Find the k nearest neighbors based on cosine similarity
        
        distances, indices = knn.kneighbors(movie_vector)

        # Get the movieIds of the nearest neighbors
        nearest_movie_ids = movie_features.iloc[indices[0]]['movieId'].values
    # print(nearest_movie_ids)
        # Get ratings from the user for these nearest neighbors
        
        user_ratings = train_df[
            (train_df['userId'] == user_id) &
            (train_df['movieId'].isin(nearest_movie_ids))
        ]['rating']
        
        if not user_ratings.empty:
            fallback_counter['knn_ratings'] += 1
            return user_ratings.mean()
        

        # Fallback to user's global mean if no ratings are available for nearest neighbors
        user_ratings = train_df[train_df['userId'] == user_id]['rating']
        if not user_ratings.empty:
            fallback_counter['user_mean'] += 1
            return user_ratings.mean()

        # Fallback to global mean if no ratings are available for the user
        fallback_counter['global_mean'] += 1
        return train_df['rating'].mean()

    # Predict on test set
    from tqdm import tqdm

    test_df['pred_rating'] = np.nan  # Initialize the 'pred_rating' column
    for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting Ratings, standby..."):
        
        test_df.at[index, 'pred_rating'] = predict_user_rating_knn(row['userId'], row['movieId'])


    test_eval = test_df.dropna(subset=['pred_rating'])

    rmse = np.sqrt(mean_squared_error(test_eval['rating'], test_eval['pred_rating']))
    mae = mean_absolute_error(test_eval['rating'], test_eval['pred_rating'])

    print(f"\nKNN-Based Prediction — RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    if plot:
        plot_ratings_comparison(test_eval['rating'], test_eval['pred_rating'])
        plot_neighbor_distance_distribution(knn, features_cosine)

    # Fallback statistics
    total_preds = sum(fallback_counter.values())
    print("\nFallback Statistics:")
    for method, count in fallback_counter.items():
        print(f"  {method}: {count} ({count / total_preds:.2%})")
    
    return rmse, mae
 
def plot_rmse_mae_by_k(k_values, rmse, mae):
    plt.figure(figsize=(12, 6))

    # Plot the RMSE and MAE
    plt.plot(k_values, rmse, marker='o', label='RMSE')
    plt.plot(k_values, mae, marker='o', label='MAE')

    # Set log scale for x-axis since k_values span a wide range
    plt.xscale('log', base=2)  # Log scale for the x-axis

    # Set log-spaced ticks for the x-axis
    plt.xticks(k_values)  # Set the specific k_values as the ticks
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Error')
    plt.title('RMSE and MAE vs Number of Neighbors (k) (Log Scale)')

    # Add grid, legend, and layout
    plt.legend()
    plt.grid(True, which="both", ls="--")  # Both major and minor grid lines
    plt.tight_layout()

    # Save and display the plot
    plt.savefig("visuals/rmse_mae_by_k.png")
    plt.show()



if __name__=='__main__':


    import argparse
    parser = argparse.ArgumentParser(description='KNN Recommender System')
    parser.add_argument('--hyperparameter', action='store_true')
    args = parser.parse_args()

    if args.hyperparameter:
        k_values = [1,2,4,8,16,32,64,128,256,512,1024]
        rmses = []
        maes = []

        for k in k_values:
            print(f"--------Training KNN with k={k}...--------")
            rmse, mae= train(k, plot=False)
            rmses.append(rmse)
            maes.append(mae)
            print('-'*50)
        plot_rmse_mae_by_k(k_values, rmses, maes)
    else:
        train(k=500, plot=True)
    


