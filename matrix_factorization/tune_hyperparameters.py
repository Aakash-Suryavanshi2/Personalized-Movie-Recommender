import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from model import MatrixFactorizationModel
from utils.matrix_utils import split_matrix

def tune_num_features(R, num_features_list, alpha = 0.002, beta = 0.02, epochs = 50, seed = 42):
    num_users, num_items = R.shape
    train_R, test_R = split_matrix(R, test_ratio=0.2, seed=seed)
    results = []

    for num_features in num_features_list:
        print(f"\nTraining with num_features = {num_features}")

        model = MatrixFactorizationModel(
            num_users=num_users,
            num_items=num_items,
            num_features=num_features,
            alpha=alpha,
            beta=beta,
            seed=seed
        )
        model.train(train_R, epochs=epochs)

        predicted_ratings = model.predict()
        test_mask = test_R > 0
        actual = test_R[test_mask]
        predicted = predicted_ratings[test_mask]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        print(f"Final RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        results.append({
            'num_features': num_features,
            'rmse': rmse,
            'mae': mae
        })

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    plt.plot(results_df['num_features'], results_df['rmse'], marker='o', linestyle='-')
    plt.title("RMSE vs. Number of Latent Features")
    plt.xlabel("Number of Features")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.savefig(f"rmse_vs_features.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df['num_features'], results_df['mae'], marker='o', linestyle='-', color='orange')
    plt.title("MAE vs. Number of Latent Features")
    plt.xlabel("Number of Features")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.savefig(f"mae_vs_features.png")
    plt.close()

def tune_alpha(R, alpha_list, num_features = 100, beta = 0.02, epochs = 50, seed = 42):
    num_users, num_items = R.shape
    train_R, test_R = split_matrix(R, test_ratio=0.2, seed=seed)
    results = []

    for alpha in alpha_list:
        print(f"\nTraining with alpha = {alpha}")

        model = MatrixFactorizationModel(
            num_users=num_users,
            num_items=num_items,
            num_features=num_features,
            alpha=alpha,
            beta=beta,
            seed=seed
        )
        model.train(train_R, epochs=epochs)

        predicted_ratings = model.predict()
        test_mask = test_R > 0
        actual = test_R[test_mask]
        predicted = predicted_ratings[test_mask]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        print(f"Final RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        results.append({
            'alpha': alpha,
            'rmse': rmse,
            'mae': mae
        })

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    plt.plot(results_df['alpha'], results_df['rmse'], marker='o', linestyle='-')
    plt.title("RMSE vs. Learning Rate (alpha)")
    plt.xlabel("Alpha")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.savefig(f"rmse_vs_alpha.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df['alpha'], results_df['mae'], marker='o', linestyle='-', color='orange')
    plt.title("MAE vs. Learning Rate (alpha)")
    plt.xlabel("Alpha")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.savefig(f"mae_vs_alpha.png")
    plt.close()

def tune_beta(R, beta_list, num_features = 100, alpha = 0.002, epochs = 50, seed = 42):
    num_users, num_items = R.shape
    train_R, test_R = split_matrix(R, test_ratio=0.2, seed=seed)
    results = []

    for beta in beta_list:
        print(f"\nTraining with beta = {beta}")

        model = MatrixFactorizationModel(
            num_users=num_users,
            num_items=num_items,
            num_features=num_features,
            alpha=alpha,
            beta=beta,
            seed=seed
        )
        model.train(train_R, epochs=epochs)

        predicted_ratings = model.predict()
        test_mask = test_R > 0
        actual = test_R[test_mask]
        predicted = predicted_ratings[test_mask]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        print(f"Final RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        results.append({
            'beta': beta,
            'rmse': rmse,
            'mae': mae
        })

    results_df = pd.DataFrame(results)
    import os
    os.chdir(__file__)
    os.chdir('visuals/hyperparameters')
    
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['beta'], results_df['rmse'], marker='o', linestyle='-')
    plt.title("RMSE vs. Regularization Strength (beta)")
    plt.xlabel("Beta")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.savefig(f"rmse_vs_beta.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(results_df['beta'], results_df['mae'], marker='o', linestyle='-', color='orange')
    plt.title("MAE vs. Regularization Strength (beta)")
    plt.xlabel("Beta")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.savefig(f"mae_vs_beta.png")
    plt.close()

def tune_epochs(R, max_epochs, num_features = 100, alpha = 0.002, beta = 0.04, seed = 42):
    num_users, num_items = R.shape
    train_R, test_R = split_matrix(R, test_ratio=0.2, seed=seed)
    test_mask = test_R > 0

    model = MatrixFactorizationModel(
        num_users=num_users,
        num_items=num_items,
        num_features=num_features,
        alpha=alpha,
        beta=beta,
        seed=seed
    )

    losses = []
    rmse_by_epoch = []
    mae_by_epoch = []
    epoch_checkpoints = []

    print("\nBeginning epoch tuning...\n")
    for epoch in range(1, max_epochs + 1):
        model.train(train_R, epochs=1)
        predicted_ratings = model.predict()

        train_mask = train_R > 0
        error_matrix = train_R - predicted_ratings
        total_loss = np.sum((error_matrix[train_mask]) ** 2)
        reg_term = beta * (np.sum(model.U**2) + np.sum(model.V**2))
        loss = total_loss + reg_term
        losses.append(loss)

        if epoch % 10 == 0:
            actual = test_R[test_mask]
            predicted = predicted_ratings[test_mask]
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)
            rmse_by_epoch.append(rmse)
            mae_by_epoch.append(mae)
            epoch_checkpoints.append(epoch)
            print(f"Final RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_epochs + 1), losses, linestyle='-')
    plt.title("Training Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"loss_vs_epoch.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_checkpoints, rmse_by_epoch, marker='o', linestyle='-')
    plt.title("RMSE vs. Epoch (Test Set)")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"rmse_vs_epoch.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_checkpoints, mae_by_epoch, marker='o', linestyle='-', color='orange')
    plt.title("MAE vs. Epoch (Test Set)")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"mae_vs_epoch.png")
    plt.close()