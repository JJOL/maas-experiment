import wandb
import sys

import torch
import random
import numpy as np

from train import run_experiment

if __name__ == "__main__":
    RD_SEED = random.randint(0,100000)
    random.seed(RD_SEED)
    np.random.seed(RD_SEED)
    torch.manual_seed(RD_SEED)
    
    default_config = {
        "path": f"training/{sys.argv[1]}_token",
        "token_encoding": "Octuple",
        # "token_encoding": "Octuple_Multinstrument",
        "model_type": "GRU", # [RNN, GRU, LSTM],
        "preprocess": 'Normalized_Significant_Features', # -> 3 first columns of Octuple
        "sequence_len": 77,##50
        "hidden_size": 256, ##256
        "rnn_layers": 2, ##4
        "rnn_dropout": 0.25,
        "fcl_dropout": 0.4,
        "rd_seed": RD_SEED,
        "split_ratio": 0.826838, ##90
        "batch_size": 123, ##64
        "learning_rate": 0.0012399, ##0.005
        "epochs": 80, ##100
        "loss_function": "NLL",
        "optimizer": "Adam" #[SGD, Adam]
    }

    wandb_run = wandb.init(config=default_config,
        project="maas-genres",
        entity="symbolic-music-patterns-props",
        notes="Experiment X Config 1",
        tags=["genres10", "discovery", "low_transcription", "low_classification"])
    run_config = wandb.config

    wandb.log({ "rand_seed": RD_SEED })
    run_experiment(run_config, wandb_run, play=False)
