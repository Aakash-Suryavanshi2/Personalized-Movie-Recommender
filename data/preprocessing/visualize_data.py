import pandas as pd
import matplotlib.pyplot as plt

def visualize_movie_ratings(df_path):
    df = pd.read_csv(df_path)
    rating_counts = df['rating'].value_counts().sort_index()
    print("Rating Distribution:")
    print(rating_counts)
    plt.figure(figsize=(8, 6))
    rating_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel("Rating")
    plt.ylabel("Number of Movies")
    plt.title("Distribution of Movie Ratings")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("visuals/movie_rating_distribution.png")
    plt.show()

def visualize_genres(df_path):  
    df = pd.read_csv(df_path)
    if 'genres' not in df.columns:
        raise ValueError("The 'genres' column is missing from the dataset.")
    df['genres'] = df['genres'].fillna("Unknown")
    df['genre_list'] = df['genres'].str.split(r'\|')
    df_exploded = df.explode('genre_list')
    df_exploded['genre_list'] = df_exploded['genre_list'].str.strip()
    genre_counts = df_exploded['genre_list'].value_counts().sort_index()
    print("Genre Distribution:")
    print(genre_counts)
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar', color='mediumseagreen', edgecolor='black')
    plt.xlabel("Genre")
    plt.ylabel("Number of Movies")
    plt.title("Distribution of Movies by Genre")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visuals/genre_distribution.png")
    plt.show()

def visualize_user_ratings(df_path):
    ratings_df = pd.read_csv(df_path)
    required_cols = ['userId', 'movieId', 'rating']
    for col in required_cols:
        if col not in ratings_df.columns:
            raise ValueError(f"Column '{col}' is missing from the ratings dataset.")
    user_rating_counts = ratings_df.groupby("userId")["movieId"].count().reset_index()
    user_rating_counts.columns = ["userId", "num_ratings"]
    user_rating_counts = user_rating_counts.sort_values(by="num_ratings", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    plt.bar(user_rating_counts.index,
            user_rating_counts["num_ratings"],
            align='edge',
            color="skyblue", edgecolor="black")
    plt.xlim(0, len(user_rating_counts))

    plt.xlabel("User Rank (sorted by number of ratings)")
    plt.ylabel("Number of Movies Rated")
    plt.title("Number of Movies Rated per User (Ranked)")
    plt.tight_layout()
    plt.savefig("visuals/user_rating_distribution.png")
    plt.show()


if __name__ == '__main__':
    import os
    DATA_PATH = os.path.join(__file__,'../../')
    
    os.chdir(DATA_PATH)
    

    visualize_movie_ratings("data.csv")

    visualize_genres("movies.csv")

    visualize_user_ratings("ratings.csv")

