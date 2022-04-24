import wandb
import sys

from train import run_experiment

if __name__ == "__main__":
    dataset_params = {
        "path": f"training/{sys.argv[1]}_token"
    }
    preprocessing_params = {
        "token_encoding": "Octuple"
    }
    model_params = {
        "model_type": "GRU", # [RNN, GRU, LSTM],
        "preprocess": 'Normalized_Significant_Features', # -> 3 first columns of Octuple
        "sequence_len": 50,
        "hidden_size": 256,
        "rnn_layers": 4
    }
    training_params = {
        "split_ratio": 0.90,
        "batch_size": 64,
        "learning_rate": 0.005,
        "epochs": 100,
        "loss_function": "NLL",
        "optimizer": "Adam" #[SGD, Adam]
    }

    default_config = {}
    for k in dataset_params:
        default_config[k] = dataset_params[k]
    for k in preprocessing_params:
        default_config[k] = preprocessing_params[k]
    for k in model_params:
        default_config[k] = model_params[k]
    for k in training_params:
        default_config[k] = training_params[k]

    wandb_run = wandb.init(config=default_config, 
        project="maas-genres",
        entity="symbolic-music-patterns-props",
        notes="Explore optimal RNN hyperparameters",
        tags=["baseline01", "gtzan_genres", "discovery"])
    run_config = wandb.config

    run_experiment(run_config, wandb_run, play=False)