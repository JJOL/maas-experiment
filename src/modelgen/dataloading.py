from os import listdir
import math
import random
import torch
from torch.utils.data import Dataset

def get_all_data_names(dataset_path):
    data = []
    categories = listdir(dataset_path)
    for cat in categories:
        data = data + listdir(f"{dataset_path}/{cat}")
    return data



def _padded_list(data, data_length, input_size):
    fillin = [[0]*input_size]*(data_length-len(data))
    data = data + fillin
    return data

def _make_x_mask(length, position, input_size):
    zero = [0]*input_size
    one  = [1]*input_size
    mask = [zero]*length
    mask[position-1] = one
    return mask

def _read_music_tensor(file_path, max_token_len, input_size, hidden_size):
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
        padded = torch.tensor(_padded_list(token_data, max_token_len, input_size))
        mask = torch.tensor(_make_x_mask(max_token_len, len(token_data), hidden_size))
        
        return [torch.cat((padded, mask), dim=1)]
    else:
        segments = []
        n = math.ceil(len(token_data) / max_token_len)
        for i in range(n):
            inf = i*max_token_len
            sup = inf + max_token_len
            segment_token_data = token_data[inf:sup]

            padded = torch.tensor(_padded_list(segment_token_data, max_token_len, input_size))
            mask = torch.tensor(_make_x_mask(max_token_len, len(segment_token_data), hidden_size))
            segments.append(torch.cat((padded, mask), dim=1))
        return segments

def _cat_index_tensor(category, categories):
    return categories.index(category)

def get_data_splitted(dataset_path, sequence_len, split_ratio, input_size, hidden_size):
    data = []
    categories = listdir(dataset_path)
    for cat in categories:
        for sample in listdir(f"{dataset_path}/{cat}"):
            segments = _read_music_tensor(f"{dataset_path}/{cat}/{sample}", sequence_len, input_size, hidden_size)
            label_tensor = _cat_index_tensor(cat, categories)
            for token_tensor in segments:
                data.append((token_tensor, label_tensor))

    # Shuffle Data
    random.shuffle(data)

    training_n = math.floor(len(data)*split_ratio)
    train_data = data[:training_n]
    test_data = data[training_n:]

    return train_data, test_data

def _get_all_labeled_names(dataset_path):
    labeled_data = []
    labels = listdir(dataset_path)
    for cat in labels:
        samples = listdir(f"{dataset_path}/{cat}")
        cats = [cat]*len(samples)
        labeled_data += list(zip(samples, cats))
    return labeled_data

def _select_items(full_list, indices):
    return list(map(full_list.__getitem__, indices))


def get_data_by_kgroup(dataset_path, sequence_len, input_size, hidden_size, train_indices, test_indices):
    combined_data = []
    categories = listdir(dataset_path)
    classified_filenames = _get_all_labeled_names(dataset_path)

    combined_classified_filednames = [_select_items(classified_filenames, train_indices), _select_items(classified_filenames, test_indices)]
    for filenames in combined_classified_filednames:
        data = []
        for fileName, cat in filenames:
            segments = _read_music_tensor(f"{dataset_path}/{cat}/{fileName}", sequence_len, input_size, hidden_size)
            label_tensor = _cat_index_tensor(cat, categories)
            for token_tensor in segments:
                data.append((token_tensor, label_tensor))
        combined_data.append(data)

    return combined_data[0], combined_data[1]

# def get_column_reduced(train_data, test_data):
#     # Good Cols: [Pitch: 0, Velocity: 1, Duraction: 2] // Rest: [Program: 3, Position: 4, Bar: 5]
#     def cut_insignifant_cols(xy):
#         x = xy[0]
#         print(f"To Reduce X Size: {x.size()}")
#         trimmed = torch.tensor([x[0].item(), x[1].item(), x[2].item()])
#         return (trimmed, xy[1])
#     train_data = list(map(cut_insignifant_cols, train_data))
#     test_data = list(map(cut_insignifant_cols, test_data))

#     return train_data, test_data


# def get_normalized_data(train_data, test_data):
#     def empiric_normalize(xy):
#         x = xy[0]
#         n_pitch = (x[0].item() - 2) / (86 - 2)
#         n_vel   = (x[1].item() - 116) / (120 - 116)
#         n_dur   = (x[2].item() - 121) / (184 - 121)
#         return (torch.tensor([n_pitch, n_vel, n_dur]), xy[1])

#     train_data = list(map(empiric_normalize, train_data))
#     test_data = list(map(empiric_normalize, test_data))

    return train_data, test_data

class TokenFileDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]
