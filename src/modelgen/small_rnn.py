import torch.nn as nn
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_type, n_layers):
        super(TorchRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = n_layers

        if model_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, self.num_layers, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True)
        elif model_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, self.num_layers, batch_first=True)

        self.fcl = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        # N = x.size()[0]
        # h0 = torch.zeros(self.num_layers, N, self.hidden_size).to(device)
        out, _ = self.rnn(x)
        # out = [N, L, H_size=128]
        # out = torch.sum(out * x_mask, dim=1)
        out = out[:, -1, :]
        # out = [N, Hout]
        out = self.fcl(out)
        out = self.softmax(out)
        return out


class ManualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManualRNN, self).__init__()

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