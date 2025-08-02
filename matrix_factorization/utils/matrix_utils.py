import numpy as np

def split_matrix(R, test_ratio=0.2, seed=42):
    np.random.seed(seed)
    train = R.copy()
    test = np.zeros_like(R)
    for user in range(R.shape[0]):
        rated_items = np.where(R[user] > 0)[0]
        if len(rated_items) == 0:
            continue
        test_size = max(1, int(len(rated_items) * test_ratio))
        test_items = np.random.choice(rated_items, size=test_size, replace=False)
        train[user, test_items] = 0
        test[user, test_items] = R[user, test_items]
    return train, test

def display_top_k_recommendations(user_index, predicted_ratings, train_R, movie_ids, movies_df, user_item_matrix, K=5):
    user_ids = user_item_matrix.index.tolist()
    movie_id_lookup = {i: movie_ids[i] for i in range(len(movie_ids))}
    movie_info = movies_df.set_index('movieId')

    user_id = user_ids[user_index]
    print(f"\n Top-{K} Recommendations for userId = {user_id} (user_index = {user_index})")
    print(f"{'Title':50} {'Predicted Rating':>20}")
    print("-" * 70)

    user_preds = predicted_ratings[user_index].copy()
    already_rated = train_R[user_index] > 0
    user_preds[already_rated] = -np.inf

    top_k_indices = user_preds.argsort()[::-1][:K]

    for idx in top_k_indices:
        movie_id = movie_id_lookup[idx]
        predicted_rating = user_preds[idx]
        title = movie_info.loc[movie_id]["title"] if movie_id in movie_info.index else f"Movie {movie_id}"
        print(f"{title[:48]:50} {predicted_rating:20.2f}")