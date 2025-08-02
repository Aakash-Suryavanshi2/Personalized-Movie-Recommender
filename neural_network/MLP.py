import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        
    def forward(self, x):
        x = self.model(x)
        x = x + 1
        x = x * 2.25
        x = x + 0.5
        # x range of [0.5, 5]
        return x
    