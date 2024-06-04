import torch
import os
import polars as pl
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

class LoLDatasetCache(Dataset):
    def __init__(self, max_len, n_games, data_folder, device, calculate_timestamps=False):
        self.max_len = max_len
        self.n_games = n_games
        self.cached_data = None
        self.cached_targets = None
        self.cached_file_number = -1
        self.cache_size = -1
        self.cached_timestamps = None
        self.calculate_timestamps = calculate_timestamps
        self.data_folder = data_folder
        self.device = device
    
    def __len__(self):
        return self.n_games
    
    def __getitem__(self, idx):
        file_number = int(idx // 1000)
        if self.cached_file_number != file_number:
            file_name =  f'timeline_{file_number}.parquet'
            df = pl.read_parquet(os.path.join(self.data_folder,file_name))

            grouped = df.group_by(['matchId'])
            games = []
            timestamps_per_game = []
            for _, group in grouped:
                group = group.drop('matchId')
                games.append(torch.from_numpy(group.to_numpy()))

                if self.calculate_timestamps:
                    timestamps = group['timestamp'].to_numpy()
                    max_time = timestamps[-1]
                    timestamps = (timestamps / max_time).astype(np.float32)
                    timestamps = (timestamps * 100).astype(np.int32)
                    timestamps_per_game.append(torch.from_numpy(timestamps))
            
            games = pad_sequence(games, batch_first=True).to(torch.float)
            if games.shape[1] != self.max_len:
                padding = torch.zeros((games.shape[0], self.max_len - games.shape[1], games.shape[2]))
                games = torch.cat((games, padding), 1)

            if self.calculate_timestamps:
                timestamps_per_game = pad_sequence(timestamps_per_game, batch_first=True, padding_value=100).to(torch.int).to(self.device)
                if timestamps_per_game.shape[1] != self.max_len:
                    padding = torch.ones((timestamps_per_game.shape[0], self.max_len - timestamps_per_game.shape[1])).to(torch.int).to(self.device) * 100
                    timestamps_per_game = torch.cat((timestamps_per_game, padding), 1)

            games[:, :, -1] = games[:, 0, -1].unsqueeze(-1).to(self.device)
            X = games[:, :, :-1]
            y = (games[:, :, -1] / 100.0 - 1).unsqueeze(-1)

            self.cached_data = X
            self.cached_targets = y
            self.cached_file_number = file_number
            self.cache_size = games.shape[0]
            self.cached_timestamps = timestamps_per_game
        
        if self.calculate_timestamps:
            return self.cached_data[idx % 1000], self.cached_targets[idx % 1000], self.cached_timestamps[idx % 1000]
        else:
            return self.cached_data[idx % 1000], self.cached_targets[idx % 1000]
        
def index_split(n_games, seed=42):
    random.seed(seed)
    indices = np.arange(n_games)
    random.shuffle(indices)
    split_index = int(n_games // 1.1111111)
    return sorted(indices[:split_index]),sorted(indices[split_index:])

def get_loaders(max_len, n_games, data_folder, device, calculate_timestamps = False, batch_size = 6):
    dataset = LoLDatasetCache(max_len, n_games, data_folder, device, calculate_timestamps=calculate_timestamps)
    train_indices, test_indices = index_split(n_games)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader