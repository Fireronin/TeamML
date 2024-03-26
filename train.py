#%%
from transformer import TransformerModel

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
import polars as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import plotly.graph_objects as go
import plotly.express as px
import lovely_tensors as lt
import numpy as np

lt.monkey_patch()
EPOCHS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
BATCH_SIZE = 8

print(f'Device: {DEVICE}')


# 58 per player, 48 + 1 + 9
# 227 items (?)
# 70 runes (?)
output_dim = 1
nhead = 10
nlayers = 2
ngame_cont = 120
nteam_cont = 0
nplayer_cont = 48
nitems = 22700
nchampions = 1670
nrunes = 70000
game_dim = 50
team_dim = 0
player_dim = 30
item_dim = 20
champion_dim = 30
runes_dim = 5
dropout = 0.1


class LoLDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

df = pl.read_parquet('transformed_data/timeline_0.parquet')

# for each column compute the number of unique values, mean, std and max
n_unique = []
mean = []
std = []
for col in df.columns:
    if col == 'matchId':
        continue
    n_unique.append(df[col].n_unique())
    mean.append(df[col].mean())
    std.append(df[col].std())
    
# if std is 0, replace it with 1
std = [s if s != 0 else 1 for s in std]

games = []
grouped = df.group_by(['matchId'])
# print number of games
print(f'Number of games: {grouped.count().shape[0]}')

for name, group in grouped:
    group = group.drop('matchId')
    games.append(torch.from_numpy(group.to_numpy()))

games = pad_sequence(games, batch_first=True).to(torch.float)
games[:, :, -1] = games[:, 0, -1].unsqueeze(-1)
X = games[:, :, :-1]
y = (games[:, :, -1] / 100.0 - 1).unsqueeze(-1)
dataset = LoLDataset(X, y)
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

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
    n_unique,
    mean,
    std,
    dropout
).to(DEVICE)

# print number of parameters
print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

optimizer = optim.Adam(model.parameters(),lr=0.01)
criterion = nn.BCEWithLogitsLoss()
#%%
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

    # print the loss
    # print(f'Epoch: {epoch}, Loss: {torch.mean(torch.tensor(losses))}')

max_game_len = X.shape[1]

accuracy_per_timestep = np.zeros(max_game_len)
added_preds = np.zeros(max_game_len)

model.eval()

for X, y in test_loader:
    X = X.to(DEVICE)
    y = y.to(DEVICE)

    y = y.squeeze().detach().cpu().numpy()
    y_pred = model(X).squeeze().detach().cpu().numpy()
    X = X.detach().cpu().numpy()

    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = 0 

    game_len = np.where(X[:, :,-1] > 0)[1][-1] + 1
    accuracy = (y_pred == y)
    accuracy_per_timestep[:game_len] += accuracy[:game_len]
    added_preds[:game_len] += 1
    

accuracy_per_timestep = accuracy_per_timestep / added_preds

trace0 = go.Scatter(
    y=accuracy_per_timestep,
    mode='lines',
    name='accuracy'
)

added_preds_normalized = added_preds / added_preds.max()

trace1 = go.Scatter(
    y=added_preds_normalized,
    mode='lines',
    name='added_preds'
)

fig = go.Figure(data=[trace0,trace1])
fig.show()