import os
from os import listdir
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np

from dataloading import get_all_data_names, get_data_by_kgroup, TokenFileDataset, _select_items

from small_rnn import TorchRNN
from metrics import classification_scores

import wandb

def _make_new_dir(target_path):
    if not os.path.isdir(target_path):
        os.mkdir(target_path)

K_FOLDS = 5

host_device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU CUDA")
else:
    print("Using CPU!")

def run_experiment(run_config, wandb_run, play=False):
    input_size = 3 if run_config["token_encoding"] == "Octuple" else 10
    run_config["input_size"] = input_size
    classes = listdir(run_config["path"])
    n_classes = len(classes)

    cross_val_metrics = {
        'mean_average_f1': 0.0,
        'mean_average_precision': 0.0
    }

    _make_new_dir("metrics_output")

    all_dataset_names = get_all_data_names(run_config["path"])
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=run_config["rd_seed"])
    for k_index, (train_index, test_index) in enumerate(kf.split(all_dataset_names)):
        print(f"Fold {k_index+1}#:")
        # freq_table = {
        #     "train_Positive": 0,
        #     "train_Negative": 0,
        #     "test_Positive": 0,
        #     "test_Negative": 0
        # }
        # for train_samp in _select_items(all_dataset_names, train_index):
        #     if train_samp.startswith("_Whats_Going"):
        #         freq_table["train_Positive"] += 1
        #     else:
        #         freq_table["train_Negative"] += 1
        # for train_samp in _select_items(all_dataset_names, test_index):
        #     if train_samp.startswith("_Whats_Going"):
        #         freq_table["test_Positive"] += 1
        #     else:
        #         freq_table["test_Negative"] += 1

        # print(f"|Train P\tTrain N\n|{freq_table['train_Positive']}\t{freq_table['train_Negative']}")
        # print(f"|Test P\tTest N\n|{freq_table['test_Positive']}\t{freq_table['test_Negative']}")
        print("Creating model....")
        model = create_model(run_config, input_size, n_classes)

        train_dataloader, test_dataloader = create_dataloader(run_config, train_index, test_index, input_size)
        # if k_index != 4:
        #     continue
        train_model(model, train_dataloader, test_dataloader, run_config, k_index)
        metrics = report_results(test_dataloader, model, classes, k_index, run_config["input_size"])
        cross_val_metrics["mean_average_f1"] += (metrics["average_f1"] / K_FOLDS)
        cross_val_metrics["mean_average_precision"] += (metrics["average_precision"] / K_FOLDS)
        
    print(f"Mean Average F1: {cross_val_metrics['mean_average_f1']}, Mean Average Precision: {cross_val_metrics['mean_average_precision']}")
    wandb.log(cross_val_metrics)
    wandb_run.finish()


    #     play_model(play, run_config, input_size, model, classes)

# def play_model(play, run_config, input_size, model, classes):
#     # 5.1 Play with model?
#     while play:
#         cmd = str(input('Insert music token path or quit: '))
#         if cmd == 'quit':
#             break
        
#         seqs_list = _read_music_tensor(cmd, run_config["sequence_len"], input_size)
#         seqs = torch.cat(seqs_list).view(len(seqs_list), 50, input_size).to(device)
#         # seq list([50, Hin])
#         with torch.set_grad_enabled(False):
#             outputs = model(seqs)
#             accumulated_output = torch.sum(outputs, dim=0)
#             accumulated_output = accumulated_output.view(1, len(classes))
#             _, predicted = torch.max(accumulated_output, dim=1)

#             print(f"The predicted class is: {classes[predicted.item()]}")

"""
create_dataloader()
"""
def create_dataloader(config, train_index, test_index, input_size):
    """Returns train and test pytorch.Datasets and respective pytorch.Dataloaders in form (train_dataset, train_dataloader, test_dataset, test_dataloader)"""
    train_data, test_data = get_data_by_kgroup(config['path'], config['sequence_len'], input_size, config['hidden_size'], train_index, test_index)

    train_dataloader =  DataLoader(TokenFileDataset(train_data), batch_size=config['batch_size'], shuffle=True)
    test_dataloader =  DataLoader(TokenFileDataset(test_data), shuffle=True, batch_size=min(128, len(test_data)))

    return train_dataloader, test_dataloader
    
def create_model(model_params, input_size, output_size):
    model = TorchRNN(input_size, model_params["hidden_size"], output_size, model_params["model_type"], model_params["rnn_layers"], model_params)
    model = model.to(device)
    return model

def _smooth_filter(values_arr):
    smoothed_arr = []
    for i in range(len(values_arr)):
        prev = values_arr[i-1] if i > 0 else values_arr[i]
        curr = values_arr[i]
        succ = values_arr[i+1] if i < (len(values_arr) - 1) else values_arr[i]
        v = (2*prev + 3*curr + 2*succ) / 7
        smoothed_arr.append(v)
    return smoothed_arr


def train_model(model, train_dataloader, test_dataloader, training_params, kfold):
    print("Training Model...\n\n")
    lr = training_params["learning_rate"]
    input_size = training_params["input_size"]
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
            labels = model(data_batch[:,:,:input_size], data_batch[:,:,input_size:])
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
                outputs = model(data_batch[:,:,:input_size], data_batch[:,:,input_size:])
                # print(f"Val Output Size: {outputs.size()}")
                # print(f"Val Batch outputs size: {label_batch.size()}")

                loss = criterion(outputs, label_batch)
                validation_loss += loss.item()

                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, dim=1)
                n_samples += outputs.size(0)
                n_correct += (predicted == label_batch).sum().item()
            
            validation_loss = validation_loss / len(test_dataloader)

            acc = 100.0 * n_correct / n_samples
            all_accuracies.append(acc)
            print(f'Accuracy of the network on test: {acc} %')

        print(f"Training Loss: {training_loss} Validation Loss: {validation_loss}")
        all_validation_losses.append(validation_loss)
        wandb.log({f"k{kfold}_training_loss": training_loss, f"k{kfold}_validation_loss": validation_loss, f"k{kfold}_accuracy": acc})

    plt.figure()
    plt.title('Model Loss')
    plt.plot(all_training_losses, 'r')
    plt.plot(all_validation_losses, 'b')
    plt.legend(['Training', 'Validation'])
    plt.savefig(f'metrics_output/k{kfold}_model_losses.jpg')

    smoothed_accuracies = _smooth_filter(all_accuracies)

    plt.figure()
    plt.title('Model Accuracy')
    plt.plot(all_accuracies, 'r')
    plt.plot(smoothed_accuracies, 'b--')
    plt.savefig(f'metrics_output/k{kfold}_model_accuracy.jpg')

    acc_fluctuation_err = np.array(smoothed_accuracies) - np.array(all_accuracies)
    acc_fluct_err = np.sum(acc_fluctuation_err ** 2) / len(smoothed_accuracies)
    wandb.log({f'k{kfold}_acc_vtmk': acc_fluct_err})

    smoothed_val_losses = _smooth_filter(all_validation_losses)
    valloss_fluctuation_err = np.array(smoothed_val_losses) - np.array(all_validation_losses)
    valloss_fluct_err = np.sum(valloss_fluctuation_err ** 2) / len(smoothed_val_losses)
    wandb.log({f'k{kfold}_val_loss_vtmk': valloss_fluct_err})

    smoothed_train_losses = _smooth_filter(all_training_losses)
    trainloss_fluctuation_err = np.array(smoothed_train_losses) - np.array(all_training_losses)
    trainloss_fluct_err = np.sum(trainloss_fluctuation_err ** 2) / len(smoothed_train_losses)
    wandb.log({f'k{kfold}_train_loss_vtmk': trainloss_fluct_err})


    
def report_results(test_dataloader, model, classes, kfold, input_size):
    y_true = np.array([], dtype=np.int32)
    y_pred = np.array([], dtype=np.int32)
    with torch.set_grad_enabled(False):
        for feature_batch, labels_batch in test_dataloader:
            feature_batch, labels_batch = feature_batch.to(device), labels_batch.to(device)
            outputs = model(feature_batch[:,:,:input_size], feature_batch[:,:,input_size:])
            _, predicted = torch.max(outputs.data, dim=1)

            # y_true = labels_batch.to(host_device).numpy()
            # y_pred = predicted.to(host_device).numpy()
            y_true = np.concatenate((y_true, labels_batch.to(host_device).numpy()))
            y_pred = np.concatenate((y_pred, predicted.to(host_device).numpy()))

            # tn_hits = {}
            # for i in range(len(y_true)):
            #     if y_pred[i] == y_true[i]:
            #         tn_hits[y]['TP']

        
        # Absosulte Confusion Matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm_abs = pd.DataFrame(cf_matrix, index = [i for i in classes],
                            columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm_abs, annot=True)
        wandb.log({f"k{kfold}_confusion_matrix_abs": wandb.Image(plt)})
        plt.savefig(f'metrics_output/k{kfold}_conf_matrix_abs.png')

        # Relative Recall Confusion Matrix
        percentage_cf_matrix = np.array(cf_matrix, dtype=np.float32)
        totals_per_class = np.sum(cf_matrix, axis=1)
        for i in range(len(classes)):
                percentage_cf_matrix[i,:] /= totals_per_class[i]
        df_cm_rec = pd.DataFrame(percentage_cf_matrix, index = [i for i in classes],
                            columns = [i for i in classes])

        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm_rec, annot=True)
        # wandb.log({f"k{kfold}_confusion_matrix_recall": wandb.Image(plt)})
        plt.savefig(f'metrics_output/k{kfold}_conf_matrix_recall.png')

        # Relative Precision Confusion Matrix
        percentage_cf_matrix = np.array(cf_matrix, dtype=np.float32)
        totals_per_class = np.sum(cf_matrix, axis=0)
        for i in range(len(classes)):
                percentage_cf_matrix[:,i] /= totals_per_class[i]
        df_cm_prec = pd.DataFrame(percentage_cf_matrix, index = [i for i in classes],
                            columns = [i for i in classes])

        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm_prec, annot=True)
        wandb.log({f"k{kfold}_confusion_matrix_precision": wandb.Image(plt)})
        plt.savefig(f'metrics_output/k{kfold}_conf_matrix_precision.png')

        metrics = classification_scores(cf_matrix, len(classes))
        wandb.log({"average_f1": metrics["average_f1"], "average_precision": metrics["average_precision"]})
            
    return metrics