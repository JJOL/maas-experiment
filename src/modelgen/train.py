from os import listdir
import sys
import torch
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def make_run(dataset, token_encoding, model_name, model_params, training_params):

    # 1.1 Load Dataset
    classes = listdir(dataset["path"])
    print(classes)

    sequence_len = model_params["sequence_len"]
    split_ratio = training_params["split_ratio"]
    batch_size = training_params["batch_size"]
    train_data, train_dataloader, test_data, test_dataloader = create_dataloader(dataset, sequence_len, split_ratio, batch_size)

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
    input_size = 6 if token_encoding == "Octuple" else 1
    model = create_model(model_name, model_params, input_size, len(classes))
    model = model.to(device)
    # # 2.2 Compile
    
    # # 3.1 Train Model
    train_model(model, train_data, train_dataloader, test_data, test_dataloader, training_params, model_params)

    # # 4.1 Report Results (Loss, Confussion Matrix, Accuracy, Precision, F score, model version)
    # pass
import math
from torch.utils.data import Dataset, DataLoader
class TokenFileDataset(Dataset):
    def __init__(self, dataset_path, sequence_len, split_ratio, train=True):
        self.categories = listdir(dataset_path)
        self.N_CATEGORIES = len(self.categories)
        self.data = []
        for cat in self.categories:
            for sample in listdir(f"{dataset_path}/{cat}"):
                segments = self._read_music_tensor(f"{dataset_path}/{cat}/{sample}", max_token_len=sequence_len)
                label_tensor = self._onehot_tensor(cat, self.categories)
                for token_tensor in segments:
                    self.data.append((token_tensor, label_tensor))

        training_n = math.floor(len(self.data)*split_ratio)

        if train:
            self.data = self.data[:training_n]
        else:
            self.data = self.data[training_n:]

    def category_from_index(self, index):
        return self.categories[index]

    def _padded_list(self, data, data_length):
            fillin = [[0]*6]*(data_length-len(data))
            data = data + fillin
            return data

    def _read_music_tensor(self, file_path, max_token_len):
        token_data = []
        all_content = ""

        with open(file_path, "r") as file:
            all_content = file.read()

        token_lines = all_content.split('\n')
        for line in token_lines:
            if line == None or line == '':
                continue
            token_els = line.split(',')
            token_data.append([float(n) for n in token_els])
        
        if len(token_data) <= max_token_len:
            return [torch.tensor(self._padded_list(token_data, max_token_len))]
        else:
            segments = []
            n = math.ceil(len(token_data) / max_token_len)
            for i in range(n):
                inf = i*max_token_len
                sup = inf + max_token_len - 1
                segments.append(torch.tensor(self._padded_list(token_data[inf:sup], max_token_len)))
            return segments

    def _onehot_tensor(self, category, categories):
        N_CATEGORIES = len(categories)
        tensor = torch.zeros(N_CATEGORIES)
        tensor[categories.index(category)] = 1
        return tensor.long()
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

def create_dataloader(dataset, sequence_len, split_ratio, batch_size):
    train_data = TokenFileDataset(dataset['path'], sequence_len, split_ratio, train=True)
    test_data = TokenFileDataset(dataset['path'], sequence_len, split_ratio, train=False)
    train_dataloader =  DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader =  DataLoader(test_data, shuffle=True)
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

def create_model(model_name, model_params, input_size, output_size):
    model = None

    if model_name == "MusicBERT":
        pass
    else:
        model = RNN(input_size, model_params["hidden_size"], output_size)

    return model

def train_model(model, train_data, train_dataloader, test_data, test_dataloader, training_params, model_params):
    lr = training_params["lr"]
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    all_training_losses = []
    all_validation_losses = []
    n_epochs = training_params["n_epochs"]

    for i in range(n_epochs):
        print(f"Training Epoch #{i}:")
        training_loss = 0
        for j, (data_batch, label_batch) in enumerate(train_dataloader):
            if j % 10 == 0:
                print(f"Batch #{j}...")
            # Go through each x and forward
            for x_seq, y in zip(data_batch, label_batch):
                y = y.to(device)

                hidden = model.init_hidden()

                for i in range(x_seq.size()[0]):
                    x, hidden = x_seq[i].to(device), hidden.to(device)
                    output, hidden = model(x, hidden)
                    for o in output:
                        if math.isnan(o):
                            f = 2

                loss = criterion(output, y)
                training_loss += loss.item()
                if (math.isnan(training_loss)):
                    s = ""

                # Backprapogate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        training_loss /= len(train_data)

        validation_loss = 0
        with torch.set_grad_enabled(False):
            for test_batch, labe_batch in test_dataloader:
                x = test_batch[0]
                y = labe_batch[0]
                # Compute validation loss
                hidden = model.init_hidden()

                for i in range(x.size()[0]):
                    output, hidden = model(x[i], hidden)
                validation_loss += criterion(output, y).item()
            validation_loss /= len(test_data)
        print(f"Training Loss: {training_loss} Validation Loss: {validation_loss}")
        all_training_losses.append(training_loss)
        all_validation_losses.append(validation_loss)

    plt.figure()
    plt.title('Model Loss')
    plt.plot(all_training_losses, 'r')
    plt.plot(all_validation_losses, 'b')

    plt.savefig('model_losses.jpg')


if __name__ == "__main__":
    dataset_path = f"training/{sys.argv[1]}_token" 

    model_params = {
        "sequence_len": 50,
        "hidden_size": 128
    }
    training_params = {
        "split_ratio": 0.90,
        "batch_size": 1,
        "lr": 0.005,
        "n_epochs": 20
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

