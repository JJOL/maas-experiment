from cgi import test
import enum
from os import listdir
import sys
import torch
import matplotlib.pyplot as plt
import math
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np

from small_rnn import TorchRNN

import wandb
wb_run = None

host_device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU CUDA")
else:
    print("Using CPU!")
# device = torch.device('cpu')

def make_run(dataset_params, preprocessing_params, model_params, training_params, play=False):
    input_size = 3 if preprocessing_params["token_encoding"] == "Octuple" else 1
    # 1.1 Load Dataset
    # 1.2 Apply Split and Randomization
    sequence_len = model_params["sequence_len"]
    split_ratio = training_params["split_ratio"]
    batch_size = training_params["batch_size"]
    train_data, train_dataloader, test_data, test_dataloader = create_dataloader(dataset_params, sequence_len, split_ratio, batch_size, input_size)
    # 1.3 Stratification of classes?
    # 1.4 K Folds?
    # 2.1 Create model architecture
    classes = listdir(dataset_params["path"])
    model = create_model(model_params, input_size, len(classes))
    # 2.2 Compile
    model = model.to(device)
    # 3.1 Train Model
    train_model(model, train_data, train_dataloader, test_data, test_dataloader, training_params, model_params)
    # 4.1 Report Results (Loss, Confussion Matrix, Accuracy, Precision, F score, model version)
    with torch.set_grad_enabled(False):
        for feature_batch, labels_batch in test_dataloader:
            feature_batch, labels_batch = feature_batch.to(device), labels_batch.to(device)
            outputs = model(feature_batch)
            _, predicted = torch.max(outputs.data, dim=1)

            y_true = labels_batch.to(host_device).numpy()
            y_pred = predicted.to(host_device).numpy()

            cf_matrix = confusion_matrix(y_true, y_pred)
            totals_per_class = np.sum(cf_matrix, axis=1)
            # for i in len(classes):
            #     cf_matrix[i,:] /= totals_per_class[i]
            df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                                columns = [i for i in classes])
            plt.figure(figsize = (12,7))
            sn.heatmap(df_cm, annot=True)
            wandb.log({"confusion_matrix": wandb.Image(plt)})
            plt.savefig('conf_matrix.png')


    wb_run.finish()

    # 5.1 Play with model?
    while play:
        cmd = str(input('Insert music token path or quit: '))
        if cmd == 'quit':
            break
        
        seqs_list = _read_music_tensor(cmd, model_params["sequence_len"], input_size)
        seqs = torch.cat(seqs_list).view(len(seqs_list), 50, input_size).to(device)
        # seq list([50, Hin])
        with torch.set_grad_enabled(False):
            outputs = model(seqs)
            accumulated_output = torch.sum(outputs, dim=0)
            accumulated_output = accumulated_output.view(1, len(classes))
            _, predicted = torch.max(accumulated_output, dim=1)

            print(f"The predicted class is: {classes[predicted.item()]}")

def _padded_list(data, data_length, input_size):
    fillin = [[0]*input_size]*(data_length-len(data))
    data = data + fillin
    return data

def _read_music_tensor(file_path, max_token_len, input_size):
    token_data = []
    all_content = ""

    with open(file_path, "r") as file:
        all_content = file.read()

    token_lines = all_content.split('\n')
    for line in token_lines:
        if line == None or line == '':
            continue
        token_els = line.split(',')
        token_els = token_els[0:3] #TODO Put in another place column trim filter
        token_els = [float(n) for n in token_els]
        n_pitch = (token_els[0] - 2) / (86 - 2) #TODO Put in another place normalization
        n_vel   = (token_els[1] - 116) / (120 - 116)
        n_dur   = (token_els[2] - 121) / (184 - 121)
        token_els = [n_pitch, n_vel, n_dur]
        token_data.append(token_els)
    
    if len(token_data) <= max_token_len:
        return [torch.tensor(_padded_list(token_data, max_token_len, input_size))]
    else:
        segments = []
        n = math.ceil(len(token_data) / max_token_len)
        for i in range(n):
            inf = i*max_token_len
            sup = inf + max_token_len - 1
            segments.append(torch.tensor(_padded_list(token_data[inf:sup], max_token_len, input_size)))
        return segments

def _cat_index_tensor(category, categories):
    return categories.index(category)

def get_data_splitted(dataset_path, sequence_len, split_ratio, input_size):
    categories = listdir(dataset_path)
    data = []
    for cat in categories:
        for sample in listdir(f"{dataset_path}/{cat}"):
            segments = _read_music_tensor(f"{dataset_path}/{cat}/{sample}", sequence_len, input_size)
            label_tensor = _cat_index_tensor(cat, categories)
            for token_tensor in segments:
                data.append((token_tensor, label_tensor))

    # Shuffle Data
    random.shuffle(data)

    training_n = math.floor(len(data)*split_ratio)
    train_data = data[:training_n]
    test_data = data[training_n:]

    return train_data, test_data

def get_column_reduced(train_data, test_data):
    # Good Cols: [Pitch: 0, Velocity: 1, Duraction: 2] // Rest: [Program: 3, Position: 4, Bar: 5]
    def cut_insignifant_cols(xy):
        x = xy[0]
        print(f"To Reduce X Size: {x.size()}")
        trimmed = torch.tensor([x[0].item(), x[1].item(), x[2].item()])
        return (trimmed, xy[1])
    train_data = list(map(cut_insignifant_cols, train_data))
    test_data = list(map(cut_insignifant_cols, test_data))

    return train_data, test_data


def get_normalized_data(train_data, test_data):
    def empiric_normalize(xy):
        x = xy[0]
        n_pitch = (x[0].item() - 2) / (86 - 2)
        n_vel   = (x[1].item() - 116) / (120 - 116)
        n_dur   = (x[2].item() - 121) / (184 - 121)
        return (torch.tensor([n_pitch, n_vel, n_dur]), xy[1])

    train_data = list(map(empiric_normalize, train_data))
    test_data = list(map(empiric_normalize, test_data))

    return train_data, test_data

class TokenFileDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

"""
create_dataloader()
"""
def create_dataloader(dataset, sequence_len: int, split_ratio: float, batch_size: int, input_size: int):
    """Returns train and test pytorch.Datasets and respective pytorch.Dataloaders in form (train_dataset, train_dataloader, test_dataset, test_dataloader)"""
    train_data, test_data = get_data_splitted(dataset['path'], sequence_len, split_ratio, input_size)
    # train_data, test_data = get_column_reduced(train_data, test_data)
    # train_data, test_data = get_normalized_data(train_data, test_data)
    # Each x = [3]

    train_data = TokenFileDataset(train_data)
    test_data = TokenFileDataset(test_data)

    train_dataloader =  DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader =  DataLoader(test_data, shuffle=True, batch_size=len(test_data))

    return train_data, train_dataloader, test_data, test_dataloader
    
def create_model(model_params, input_size, output_size):
    model = TorchRNN(input_size, model_params["hidden_size"], output_size, model_params["model_type"], model_params["rnn_layers"])

    return model

def smooth_filter(values_arr):
    smoothed_arr = []
    for i in range(len(values_arr)):
        prev = values_arr[i-1] if i > 0 else values_arr[i]
        curr = values_arr[i]
        succ = values_arr[i+1] if i < (len(values_arr) - 1) else values_arr[i]
        v = (2*prev + 3*curr + 2*succ) / 7
        smoothed_arr.append(v)
    return smoothed_arr


def train_model(model, train_data, train_dataloader, test_data, test_dataloader, training_params, model_params):
    lr = training_params["learning_rate"]
    criterion = nn.NLLLoss()

    if training_params["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif training_params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    all_training_losses = []
    all_validation_losses = []
    all_accuracies = []
    n_epochs = training_params["epochs"]

    batch_limit = len(train_dataloader) // 10

    for i in range(n_epochs):
        print(f"Training Epoch #{i}:")

        training_loss = 0
        for j, (data_batch, label_batch) in enumerate(train_dataloader):
            # if j % batch_limit == 0:
            #     print(f"Batch #{j} - {j*10}%...")

            optimizer.zero_grad()
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)
            labels = model(data_batch)
            loss = criterion(labels, label_batch)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
        training_loss = training_loss / len(train_dataloader)
        all_training_losses.append(training_loss)


        validation_loss = 0
        with torch.set_grad_enabled(False):
            n_correct = 0
            n_samples = 0
            for data_batch, label_batch in test_dataloader:
                data_batch, label_batch = data_batch.to(device), label_batch.to(device)
                outputs = model(data_batch)
                # print(f"Val Output Size: {outputs.size()}")
                # print(f"Val Batch outputs size: {label_batch.size()}")

                loss = criterion(outputs, label_batch)
                validation_loss += loss.item()

                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, dim=1)
                n_samples += outputs.size(0)
                n_correct += (predicted == label_batch).sum().item()

            acc = 100.0 * n_correct / n_samples
            all_accuracies.append(acc)
            print(f'Accuracy of the network on the {len(test_data)} test sequences: {acc} %')

        print(f"Training Loss: {training_loss} Validation Loss: {validation_loss}")
        all_validation_losses.append(validation_loss)
        wandb.log({"training_loss": training_loss, "validation_loss": validation_loss, "accuracy": acc})

    plt.figure()
    plt.title('Model Loss')
    plt.plot(all_training_losses, 'r')
    plt.plot(all_validation_losses, 'b')
    plt.legend(['Training', 'Validation'])
    plt.savefig('model_losses.jpg')

    smoothed_accuracies = smooth_filter(all_accuracies)

    plt.figure()
    plt.title('Model Accuracy')
    plt.plot(all_accuracies, 'r')
    plt.plot(smoothed_accuracies, 'b--')
    plt.savefig('model_accuracy.jpg')


if __name__ == "__main__":
    dataset_params = {
        "path": f"training/{sys.argv[1]}_token"
    }
    preprocessing_params = {
        "token_encoding": "Octuple"
    }
    model_params = {
        "model_type": "RNN", # [RNN, GRU, LSTM],
        "preprocess": 'Normalized_Significant_Features', # -> 3 first columns of Octuple
        "sequence_len": 50,
        "hidden_size": 128,
        "rnn_layers": 4
    }
    training_params = {
        "split_ratio": 0.90,
        "batch_size": 64,
        "learning_rate": 0.005,
        "epochs": 130,
        "loss_function": "NLL",
        "optimizer": "SGD" #[SGD, Adam]
    }

    run_parameters = {}
    for k in dataset_params:
        run_parameters[k] = dataset_params[k]
    for k in model_params:
        run_parameters[k] = model_params[k]
    for k in training_params:
        run_parameters[k] = training_params[k]

    wb_run = wandb.init(project="maas-genres",
            entity="symbolic-music-patterns-props",
            notes="Explore optimal RNN hyperparameters",
            tags=["baseline", "binary_genres", "discovery"],
            config = run_parameters
        )
    wandb.config = run_parameters
    make_run(dataset_params, preprocessing_params, model_params, training_params, play=True)