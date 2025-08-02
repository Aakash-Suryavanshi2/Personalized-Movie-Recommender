import numpy as np
from tqdm import tqdm

class MatrixFactorizationModel:
    def __init__(self, num_users, num_items, num_features=100, alpha=0.002, beta=0.02, seed=42):
        self.num_users = num_users
        self.num_items = num_items
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        np.random.seed(seed)
        self.U = np.random.normal(scale=1./num_features, size=(num_users, num_features))
        self.V = np.random.normal(scale=1./num_features, size=(num_items, num_features))

    def train(self, R, epochs=50):
        train_mask = R > 0
        print("\nTraining Started...\n")
        for epoch in tqdm(range(epochs), desc="Training", unit="epoch", ncols=80):
            total_loss = 0
            for i in range(self.num_users):
                for j in range(self.num_items):
                    if train_mask[i, j]:
                        error = R[i, j] - np.dot(self.U[i], self.V[j])
                        self.U[i] += self.alpha * (2 * error * self.V[j] - self.beta * self.U[i])
                        self.V[j] += self.alpha * (2 * error * self.U[i] - self.beta * self.V[j])
                        total_loss += error ** 2
            reg = self.beta * (np.sum(self.U**2) + np.sum(self.V**2))

    def predict(self):
        return self.U @ self.V.T