import pandas as pd
import os
import torch
from torch.utils.data import DataLoader, Dataset
from autoencoder import Autoencoder
import numpy as np
import torch.nn as nn
## On a high level, we need two different datasets --- one for users
## and one for movies. 

# We need to train an autoencoder to learn the latent representation of each
# user, since movie embedding is pretty small.


class MoviesDataset(Dataset):
    def __init__(self, movies_path, ratings_path, merged_path, max_movies_per_batch=15):
        self.movies = pd.read_csv(movies_path) # should be pd df with userId, movieId, title, rating, genres, tags
    
        self.genre_lookup = {'Action': 0, 
                             'Mystery': 1, 
                             'Western': 2, 
                             'Thriller': 3, 
                             'Sci-Fi': 4, 
                             'Animation': 5, 
                             'Crime': 6,
                             'Romance': 7,
                             'War': 8,
                             '(no genres listed)': 9,
                             'IMAX': 10, 'Comedy': 11,
                             'Adventure': 12,
                             'Children': 13,
                             'Horror': 14,
                             'Musical': 15,
                             'Drama': 16, 
                             'Film-Noir': 17, 
                             'Fantasy': 18, 
                             'Documentary': 19}

        self.ratings = pd.read_csv(ratings_path) # should be pd df with userId, movieId, rating
        self.ratings = self.ratings.drop(columns=['timestamp'])
        self.max_movies_per_batch = max_movies_per_batch
        self.merged = pd.read_csv(merged_path) 
        self.num_users = self.ratings['userId'].nunique()
        self.user_embedding = nn.Embedding(self.num_users, 32)
        self.num_movies= self.movies['movieId'].nunique()
        print(f'number of users: {self.num_users}')
        print(f'number of movies: {self.num_movies}')
        self.movie_embedding = nn.Embedding(self.num_movies, 6)
        self.compute_movie_embeds()
        

    def __len__(self):
        return len(self.merged)

    def __getitem__(self, idx):
        movie_embed = self.movie_embeds[self.merged.iloc[idx]['movieId']]
        user_embed =  self.compute_user_embed(idx)
        rating = self.merged.iloc[idx]['rating']
        rating = torch.tensor(rating, dtype=torch.float32)
        movie_id = self.movie_id_to_index[self.merged.iloc[idx]['movieId']]
        
        user_id = self.merged.iloc[idx]['userId']
        return movie_embed, user_embed, rating, movie_id,user_id


    def compute_movie_embeds(self):
   
        ## EXAMPLE ENTRIES AND cols:
        # genres      -> only thing we can really extract is genres one hot encoded :/
        #1,1,Toy Story (1995),4.0,Adventure|Animation|Children|Comedy|Fantasy,
        ###

        #TODO maybe add a txt embedder for movie title??
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movies['movieId'].unique())}
        self.movie_embeds = {}
        for _,row in self.movies.iterrows():
            movieID = row['movieId']
            embed = torch.zeros(len(self.genre_lookup) , dtype=torch.float32) # 
            for genre in row['genres'].split('|'):
                embed[self.genre_lookup[genre]] = 1
        
            # also calculate the avg rating for each movie
            # locate all users who rated the movie, take avg
            average_rating = self.ratings[self.ratings['movieId'] == movieID]['rating'].mean()
            embed[-1] = average_rating
            movie_id_embed = self.movie_embedding(torch.tensor(self.movie_id_to_index[movieID]))
            self.movie_embeds[movieID] = embed#torch.cat((movie_id_embed,embed)).detach()
    
    
    def compute_user_embed(self, idx):
        
        user_id = self.merged.iloc[idx]['userId']
        
        # extract user id, get all movies rated w.r.t that uid
        user_idxs = self.ratings[self.ratings['userId'] == user_id].index
        user_idxs = user_idxs[user_idxs != idx]
        user_id_embed = self.user_embedding(torch.tensor(user_id - 1))  # Learnable user embedding
        
        assert self.max_movies_per_batch <= len(user_idxs)
        sampled_user_idxs = np.random.choice(user_idxs, size=self.max_movies_per_batch, replace=False)
        user_movies = self.ratings.iloc[sampled_user_idxs]
        # get max_movies_per_batch random movies for that user, excluding idx

        user_movie_ids = user_movies['movieId'].values
        user_movie_ratings = user_movies['rating'].values

        user_embed = torch.zeros(size=(self.max_movies_per_batch, self.movie_embeds[1].shape[0]+1), dtype=torch.float32)
        for i, (movie_id, movie_rating) in enumerate(zip(user_movie_ids, user_movie_ratings)):
                movie_embed = self.movie_embeds[movie_id]
                combined_embed = torch.cat((movie_embed, torch.tensor([movie_rating], dtype=torch.float32)))
                user_embed[i] = combined_embed

        return torch.cat((user_id_embed, user_embed.flatten()))
        


if __name__ == '__main__':
    DATA_PATH = os.path.join(
        os.path.dirname(__file__),
          '../data')
    os.chdir(DATA_PATH)
    movies_path = 'raw_data/movies.csv'
    ratings_path = 'raw_data/ratings.csv'
    merged_path = 'data.csv'
    dataset = MoviesDataset(movies_path, ratings_path, merged_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for movie_embed, user_embed, rating in dataloader:
        print(f"Movie Embed: {movie_embed}")
        print(f"User Embed: {user_embed}")
        print(f"Rating: {rating}")
        break