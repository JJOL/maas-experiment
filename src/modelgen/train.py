from cgi import test
from os import listdir
import sys
import torch
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def make_run(dataset, token_encoding, model_name, model_params, training_params):

    # 1.1 Load Dataset
    classes = listdir(dataset["path"])
    print(classes)

    sequence_len = model_params["sequence_len"]
    split_ratio = training_params["split_ratio"]
    batch_size = training_params["batch_size"]
    train_data, train_dataloader, test_data, test_dataloader = create_dataloader(dataset, sequence_len, split_ratio, batch_size)

    class_1 = 0
    class_2 = 0
    _, ys = next(iter(test_dataloader))
    for c in ys:
        if c == 0:
            class_1 += 1
        elif c == 1:
            class_2 += 1

    print (f"Test Dataset has {class_1} of Class 1 and {class_2} of Class 2")

    # print(f"We have {len(train_data)} training samples!")
    # print(f"We have {len(test_data)} test samples!")

    # train_features, train_labels = next(iter(test_dataloader))

    # print(f"Batch Size: {train_features.size()}")
    # x1 = next(iter(train_features))
    # y1 = next(iter(train_labels))
    # print(f"X1 Size: {x1.size()}")
    # print(f"Y1 Size: {y1.size()}")
    
    # print(f"First Label: {train_labels[0]}")
    # print(f"First Instance First Token: {train_features[0,0]}")

    # # 1.2 Apply Split and Randomization
    # # 1.3 Stratification of classes?
    # # 1.4 K Folds?

    # # 2.1 Create model architecture
    input_size = 3 if token_encoding == "Octuple" else 1
    model = create_model(model_name, model_params, input_size, len(classes))
    model = model.to(device)
    # # 2.2 Compile
    
    # # 3.1 Train Model
    train_model(model, train_data, train_dataloader, test_data, test_dataloader, training_params, model_params)

    # # 4.1 Report Results (Loss, Confussion Matrix, Accuracy, Precision, F score, model version)
    # pass
import math
import random
from torch.utils.data import Dataset, DataLoader
class TokenFileDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def category_from_index(self, index):
        return self.categories[index]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

def _padded_list(data, data_length):
    fillin = [[0]*3]*(data_length-len(data))
    data = data + fillin
    return data
    
def _read_music_tensor(file_path, max_token_len):
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
        return [torch.tensor(_padded_list(token_data, max_token_len))]
    else:
        segments = []
        n = math.ceil(len(token_data) / max_token_len)
        for i in range(n):
            inf = i*max_token_len
            sup = inf + max_token_len - 1
            segments.append(torch.tensor(_padded_list(token_data[inf:sup], max_token_len)))
        return segments

def _cat_index_tensor(category, categories):
    return categories.index(category)

def get_data_splitted(dataset_path, sequence_len, split_ratio):
    categories = listdir(dataset_path)
    data = []
    for cat in categories:
        for sample in listdir(f"{dataset_path}/{cat}"):
            segments = _read_music_tensor(f"{dataset_path}/{cat}/{sample}", max_token_len=sequence_len)
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


def create_dataloader(dataset, sequence_len, split_ratio, batch_size):
    train_data, test_data = get_data_splitted(dataset['path'], sequence_len, split_ratio)
    # train_data, test_data = get_column_reduced(train_data, test_data)
    # train_data, test_data = get_normalized_data(train_data, test_data)
    # Each x = [3]

    train_data = TokenFileDataset(train_data)
    test_data = TokenFileDataset(test_data)
    train_dataloader =  DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader =  DataLoader(test_data, shuffle=True, batch_size=len(test_data))
    return train_data, train_dataloader, test_data, test_dataloader

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 0)

        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.hidden_size)


class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super(TorchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = n_layers
        self.rnn = nn.RNN(input_size, hidden_size, self.num_layers, batch_first=True)
        self.fcl = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        N = x.size()[0]
        h0 = torch.zeros(self.num_layers, N, self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        # print(f"Out after rnn size: {out.size()}")
        # out = [N, L, H_size=128]
        out = out[:, -1, :]
        # out = [N, 1, Hout=6]
        # print(f"My Partial Output Size: {out.size()}")
        # out = [N, Hout=6]
        out = self.fcl(out)
        out = self.softmax(out)
        return out

def create_model(model_name, model_params, input_size, output_size):
    model = None

    if model_name == "MusicBERT":
        pass
    else:
        model = TorchRNN(input_size, model_params["hidden_size"], model_params["rnn_layers"], output_size)

    return model

def train_model(model, train_data, train_dataloader, test_data, test_dataloader, training_params, model_params):
    lr = training_params["lr"]
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    all_training_losses = []
    all_validation_losses = []
    all_precisions = []
    n_epochs = training_params["n_epochs"]

    for i in range(n_epochs):
        print(f"Training Epoch #{i}:")
        for j, (data_batch, label_batch) in enumerate(train_dataloader):
            print(f"Batch #{j}...")

            # print (f"Batch Data Size: {data_batch.size()}")
            data_batch, label_batch = data_batch.to(device), label_batch.to(device)
            labels = model(data_batch)

            # print(f"Output Size: {labels.size()}")
            # print(f"Batch labels size: {label_batch.size()}")

            loss = criterion(labels, label_batch)
            training_loss = loss.item() / data_batch.size()[0]
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Go through each x and forward
            # for x_seq, y in zip(data_batch, label_batch):
            #     y = y.to(device)

            #     hidden = model.init_hidden()

            #     for i in range(x_seq.size()[0]):
            #         x, hidden = x_seq[i].to(device), hidden.to(device)
            #         output, hidden = model(x, hidden)
            #         for o in output:
            #             if math.isnan(o):
            #                 f = 2

            #     loss = criterion(output, y)
            #     training_loss += loss.item()
            #     loss.backward()
            #     if (math.isnan(training_loss)):
            #         s = ""

            # # Backprapogate
            # optimizer.step()
            # optimizer.zero_grad()
        
        # training_loss /= len(train_data)

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
                validation_loss += loss.item() / data_batch.size()[0]

                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, dim=1)
                n_samples += outputs.size(0)
                n_correct += (predicted == label_batch).sum().item()

            acc = 100.0 * n_correct / n_samples
            all_precisions.append(acc)
            print(f'Accuracy of the network on the {len(test_data)} test sequences: {acc} %')


            #     x = test_batch[0]
            #     y = label_batch[0]
            #     # Compute validation loss
            #     hidden = model.init_hidden()

            #     for i in range(x.size()[0]):
            #         output, hidden = model(x[i], hidden)
            #     validation_loss += criterion(output, y).item()
            # validation_loss /= len(test_data)
        
        print(f"Training Loss: {training_loss} Validation Loss: {validation_loss}")
        all_training_losses.append(training_loss)
        all_validation_losses.append(validation_loss)

    plt.figure()
    plt.title('Model Loss')
    plt.plot(all_training_losses, 'r')
    plt.plot(all_validation_losses, 'b')
    plt.savefig('model_losses.jpg')

    plt.figure()
    plt.title('Model Precision')
    plt.plot(all_precisions)
    plt.savefig('model_precisions.jpg')


if __name__ == "__main__":
    dataset_path = f"training/{sys.argv[1]}_token" 

    model_params = {
        "sequence_len": 50,
        "hidden_size": 256,
        "rnn_layers": 4
    }
    training_params = {
        "split_ratio": 0.90,
        "batch_size": 64,
        "lr": 0.005,
        "n_epochs": 100
    }

    make_run({"path": dataset_path}, "Octuple", "custom", model_params, training_params)

# import torch
# import torch.nn as nn

# import wandb
# wandb.init(project="test-project", entity="symbolic-music-patterns-props")
# wandb.config = {
#   "learning_rate": 0.005,
#   "epochs": 6,
#   "batch_size": 1
# }

