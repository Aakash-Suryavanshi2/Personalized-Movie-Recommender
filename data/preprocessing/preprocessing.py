import pandas as pd
import os
DATA_PATH = os.path.join(__file__,'../../raw_data')    
os.chdir(DATA_PATH)


file1_path =  'movies.csv'
file2_path = 'ratings.csv'
file3_path =  'tags.csv'
output_path = 'data.csv'  

df_movies = pd.read_csv(file1_path)  # movieId, title, genres
df_ratings = pd.read_csv(file2_path)  # userId, movieId, rating, timestamp
df_tags = pd.read_csv(file3_path)  # userId, movieId, tag, timestamp


df_ratings = df_ratings.drop(columns=['timestamp'])

df_tags = df_tags.groupby(['userId', 'movieId'])['tag'].apply(list).reset_index()
df_tags['tags'] = df_tags['tag'].apply(lambda x: '|'.join(x) if x else '')
df_tags = df_tags.drop(columns=['tag'])


df_merged = pd.merge(df_ratings, df_movies, on='movieId', how='left')

df_merged = pd.merge(df_merged, df_tags, on=['userId', 'movieId'], how='left')


df_final = df_merged[['userId', 'movieId', 'title', 'rating', 'genres', 'tags']]
df_final = df_final.fillna({'tags': '', 'rating': '', 'title': '', 'genres': ''})

df_final.to_csv(output_path, index=False)

print(f"Output CSV: {output_path}")