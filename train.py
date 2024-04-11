#%%
import json
from transformer import TransformerModel
import random

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
import polars as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.utils.rnn import pad_sequence
import plotly.graph_objects as go
import plotly.express as px
import lovely_tensors as lt
import numpy as np
import os

lt.monkey_patch()

EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 6
SEED = 42

DATA_FOLDER = 'transformed_data'
GRAPHS_FOLDER = 'training_graphs'
CHECKPOINTS_FOLDER = 'checkpoints'

print(f'Device: {DEVICE}')

random.seed(SEED)

# 58 per player, 48 + 1 + 9
# 227 items (?)
# 70 runes (?)
output_dim = 1
nhead = 10
nlayers = 2
ngame_cont = 120
nteam_cont = 0
nplayer_cont = 48
nitems = 227
nchampions = 167
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
    
    def __len__(self):
        return self.n_games
    
    def __getitem__(self, idx):
        file_number = int(idx // 1000)
        if self.cached_file_number != file_number:
            file_name =  f'timeline_{file_number}.parquet'
            df = pl.read_parquet(os.path.join(DATA_FOLDER,file_name))

            grouped = df.group_by(['matchId'])
            games = []
            for _, group in grouped:
                group = group.drop('matchId')
                games.append(torch.from_numpy(group.to_numpy()))
            
            games = pad_sequence(games, batch_first=True).to(torch.float)

            if games.shape[1] != self.max_len:
                padding = torch.zeros((games.shape[0], self.max_len - games.shape[1], games.shape[2]))
                games = torch.cat((games, padding), 1)

            games[:, :, -1] = games[:, 0, -1].unsqueeze(-1)
            X = games[:, :, :-1]
            y = (games[:, :, -1] / 100.0 - 1).unsqueeze(-1)

            self.cached_data = X
            self.cached_targets = y
            self.cached_file_number = file_number
        
        return self.cached_data[idx % 1000], self.cached_targets[idx % 1000]

def index_split(n_games):
    indices = np.arange(n_games)
    random.shuffle(indices)
    split_index = int(n_games // 1.1111111)
    return sorted(indices[:split_index]),sorted(indices[split_index:])

with open('data_stats.json', 'r') as file:
    data_stats = json.load(file)

data_stats['n_games'] = 60000

dataset = LoLDatasetCache(data_stats['max_len'], data_stats['n_games'])
train_indices, test_indices = index_split(data_stats['n_games'])
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

# print number of parameters
print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

for epoch in tqdm(range(EPOCHS)):
    # set model to train mode
    model.train()

    # create a list to store the losses
    losses = []
    # loop over the data
    for X, y in tqdm(train_loader, leave=False):
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        
        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        y_pred = model(X)
        
        # compute the loss
        loss = criterion(y_pred, y)

        # append the loss to the list
        losses.append(loss.item())

        # backward pass
        loss.backward()

        # update the parameters
        optimizer.step()


    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Include any other variables that need to be saved
    }, os.path.join(CHECKPOINTS_FOLDER, f'checkpoint_{epoch}.pth'))


    mean_loss = torch.mean(torch.tensor(losses))

    trace0 = go.Scatter(
        y = losses,
        mode = 'lines',
        name = 'Loss'
    )

    trace1 = go.Scatter(
        x = [0, len(losses)-1],
        y = [mean_loss, mean_loss],
        mode = 'lines',
        name = f'Mean Loss: {mean_loss:.2f}'
    )

    fig = go.Figure(data=[trace0, trace1])
    fig.write_image(os.path.join(GRAPHS_FOLDER, 'loss', f'{epoch}.png'))


    model.eval()

    accuracy_per_timestep = np.zeros(data_stats['max_len'])
    added_preds = np.zeros(data_stats['max_len'])

    max_training_len = 0
    for X, y in test_loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        y = y.squeeze().detach().cpu().numpy()
        y_pred = model(X).squeeze().detach().cpu().numpy()
        X = X.detach().cpu().numpy()

        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = 0 

        game_len = np.where(X[:, :,-1] > 0)[1][-1] + 1
        max_training_len = max(max_training_len, game_len)
        accuracy = (y_pred == y)
        accuracy_per_timestep[:game_len] += accuracy[:game_len]
        added_preds[:game_len] += 1
        
    accuracy_per_timestep = accuracy_per_timestep[:max_training_len]
    added_preds = added_preds[:max_training_len]

    accuracy_per_timestep = accuracy_per_timestep / added_preds

    trace0 = go.Scatter(
        y=accuracy_per_timestep,
        mode='lines',
        name='Accuracy'
    )

    added_preds_normalized = added_preds / added_preds.max()

    trace1 = go.Scatter(
        y=added_preds_normalized,
        mode='lines',
        name='Number of games'
    )

    fig = go.Figure(data=[trace0,trace1])
    fig.write_image(os.path.join(GRAPHS_FOLDER, 'accuracy', f'{epoch}.png'))