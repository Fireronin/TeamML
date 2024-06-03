from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import polars as pl
import os 
import json
from transformer import TransformerModel
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import plotly.graph_objects as go

from dotenv import load_dotenv
load_dotenv()

BATCH_SIZE = 6
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'

print(f'Device: {DEVICE}')

DATA_FOLDER = 'filtered_data'
GRAPHS_FOLDER = 'evaluation_graphs'
CHECKPOINTS_FOLDER = 'checkpoints'
CHECKPOINT_FILE = 'checkpoint_5.pth'

if not os.path.exists(GRAPHS_FOLDER):
    os.makedirs(GRAPHS_FOLDER)
    
if not os.path.exists(CHECKPOINTS_FOLDER):
    os.makedirs(CHECKPOINTS_FOLDER)

random.seed(SEED)

output_dim = 1
nhead = 10
nlayers = 2
ngame_cont = 126
nteam_cont = 0
nplayer_cont = 48
nitems = 245
nchampions = 164
nrunes = 70
game_dim = 50
team_dim = 0
player_dim = 30
item_dim = 20
champion_dim = 30
runes_dim = 5
dropout = 0.1

class LoLDatasetCache(Dataset):
    def __init__(self, max_len, n_games):
        self.max_len = max_len
        self.n_games = n_games
        self.cached_data = None
        self.cached_targets = None
        self.cached_file_number = -1
        self.cache_size = -1
        self.cached_timestamps = None
    
    def __len__(self):
        return self.n_games
    
    def __getitem__(self, idx):
        file_number = int(idx // 1000)
        if self.cached_file_number != file_number:
            file_name =  f'timeline_{file_number}.parquet'
            df = pl.read_parquet(os.path.join(DATA_FOLDER,file_name))

            grouped = df.group_by(['matchId'])
            games = []
            timestamps_per_game = []
            for _, group in grouped:
                group = group.drop('matchId')
                games.append(torch.from_numpy(group.to_numpy()))

                timestamps = group['timestamp'].to_numpy()
                max_time = timestamps[-1]
                timestamps = (timestamps / max_time).astype(np.float32)
                timestamps = (timestamps * 100).astype(np.int32)
                timestamps_per_game.append(torch.from_numpy(timestamps))
            
            games = pad_sequence(games, batch_first=True).to(torch.float)
            if games.shape[1] != self.max_len:
                padding = torch.zeros((games.shape[0], self.max_len - games.shape[1], games.shape[2]))
                games = torch.cat((games, padding), 1)

            timestamps_per_game = pad_sequence(timestamps_per_game, batch_first=True, padding_value=100).to(torch.int).to(DEVICE)
            if timestamps_per_game.shape[1] != self.max_len:
                padding = torch.ones((timestamps_per_game.shape[0], self.max_len - timestamps_per_game.shape[1])).to(torch.int).to(DEVICE) * 100
                timestamps_per_game = torch.cat((timestamps_per_game, padding), 1)

            games[:, :, -1] = games[:, 0, -1].unsqueeze(-1).to(DEVICE)
            X = games[:, :, :-1]
            y = (games[:, :, -1] / 100.0 - 1).unsqueeze(-1)

            self.cached_data = X
            self.cached_targets = y
            self.cached_file_number = file_number
            self.cache_size = games.shape[0]
            self.cached_timestamps = timestamps_per_game
        
        return self.cached_data[idx % self.cache_size], self.cached_targets[idx % self.cache_size], self.cached_timestamps[idx % self.cache_size]

def index_split(n_games):
    indices = np.arange(n_games)
    random.shuffle(indices)
    split_index = int(n_games // 1.1111111)
    return sorted(indices[:split_index]),sorted(indices[split_index:])

with open('data_stats.json', 'r') as file:
    data_stats = json.load(file)

dataset = LoLDatasetCache(data_stats['max_len'], data_stats['n_games'])
train_indices, test_indices = index_split(data_stats['n_games'])
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TransformerModel(
    output_dim, 
    nhead, 
    nlayers,
    ngame_cont, 
    nteam_cont, 
    nplayer_cont, 
    nitems, 
    nchampions, 
    nrunes, 
    game_dim, 
    team_dim, 
    player_dim, 
    item_dim, 
    champion_dim, 
    runes_dim,
    data_stats['mean'],
    data_stats['std'],
    dropout
).to(DEVICE)

epoch = 0
if CHECKPOINT_FILE is not None:
    checkpoint = torch.load(os.path.join(CHECKPOINTS_FOLDER, CHECKPOINT_FILE))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']

model.eval()

accuracy_per_timestep = np.zeros(data_stats['max_len'])
# added_preds = np.zeros(data_stats['max_len'])
accuracy_per_percent = np.zeros(101)
percentage_samples = np.zeros(101)

max_training_len = 0

# X: (Batch, Game_length, Columns)
# y: (Batch, Game_length, 1)
# t: (Batch, Game_length)
for X, y, t in tqdm(test_loader):
    X = X.to(DEVICE)
    y = y.to(DEVICE)

    y = y.squeeze(-1).detach().cpu().numpy()
    y_pred = model(X).squeeze(-1).detach().cpu().numpy()
    X = X.detach().cpu().numpy()

    t = t.detach().cpu().numpy()

    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = 0 

    accuracy = (y_pred == y)


    for game_X, game_accuracy, timestamps in zip(X, accuracy, t): 
        # nonzero = np.nonzero(game_X[:,-10])
        # game_len = nonzero[-1][-1] + 1 if len(nonzero) > 0 else 0

        # max_training_len = max(max_training_len, game_len)

        # timestamps = timestamps // np.max(timestamps)

        for acc, timestamp in zip(game_accuracy, timestamps):
            accuracy_per_percent[timestamp] += acc
            percentage_samples[timestamp] += 1

        # accuracy_per_timestep[:game_len] += game_accuracy[:game_len]

        # added_preds[:game_len] += 1
    
# accuracy_per_timestep = accuracy_per_timestep[:max_training_len]
# added_preds = added_preds[:max_training_len]

accuracy_per_percent = accuracy_per_percent / percentage_samples

trace0 = go.Scatter(
    y=accuracy_per_percent,
    mode='lines',
    name='Accuracy'
)

# added_preds_normalized = added_preds / added_preds.max()

# trace1 = go.Scatter(
#     y=added_preds_normalized,
#     mode='lines',
#     name='Number of games'
# )

fig = go.Figure(data=[trace0])
if not os.path.exists(GRAPHS_FOLDER):
    os.makedirs(GRAPHS_FOLDER)
fig.write_image(os.path.join(GRAPHS_FOLDER, f'{epoch}.png'))
