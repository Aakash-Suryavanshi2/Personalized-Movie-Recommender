import wandb
from train_joint import train

sweep_config = {
    "method": "grid",  # can also be "random", "bayes"
    "metric": {
        "name": "test_loss",
        "goal": "minimize"
    },
    "parameters": {
        "num_epochs": {"value": 10},
        "lr": {"values": [1e-3]},
        "batch_size": {"values": [ 256]},
        "hidden_dim": {"values": [32, 64, 128]},
        "movie_context_len": {"values": [1,10, 19]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="movie-recommendation")
wandb.agent(sweep_id, function=train)
