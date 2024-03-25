#%%
from transformer import TransformerModel

# from dotenv import load_dotenv
# load_dotenv()

from tqdm import tqdm
import polars as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import plotly.graph_objects as go
import plotly.express as px
import lovely_tensors as lt
lt.monkey_patch()
EPOCHS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
BATCH_SIZE = 4

print(f'Device: {DEVICE}')

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
# max_ = []
for col in df.columns:
    if col == 'matchId':
        continue
    n_unique.append(df[col].n_unique())
    mean.append(df[col].mean())
    std.append(df[col].std())
    # max_.append(df[col].max())
    

# print the results
# print(f'Number of unique values: {n_unique}')
# print(f'Mean: {mean}')
# print(f'Std: {std}')
# print(f'Max: {max_}')

# if std is 0, replace it with 1
std = [s if s != 0 else 1 for s in std]

games = []
grouped = df.group_by(['matchId'])
# print number of games
print(f'Number of games: {grouped.count().shape[0]}')
# take only first 1000 games

for name, group in grouped:
    group = group.drop('matchId')
    games.append(torch.from_numpy(group.to_numpy()))

games = pad_sequence(games, batch_first=True).to(torch.float)
X = games[:, :, :-1]
y = nn.functional.one_hot((games[:, :, -1] / 100.0).to(torch.long)).to(torch.float)
dataset = LoLDataset(X, y)
X_orig, y_orig = X, y
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
# 58 per player, 48 + 1 + 9
# 227 items (?)
# 70 runes (?)
#%%
output_dim = 3
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
    # max_,
    dropout
).to(DEVICE)

# print number of parameters
print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

optimizer = optim.Adam(model.parameters(),lr=0.01)
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
#%%
for epoch in tqdm(range(EPOCHS)):
    # set model to train mode
    model.train()

    # create a list to store the losses
    losses = []
    it = 0
    # loop over the data
    for X, y in tqdm(train_loader, leave=False):
        X = X.to(DEVICE)
        y = y.to(DEVICE)
        
        # y = [4,2000,3] classes: game_finished, team_1_win, team_2_win
        
        # new_y = torch.zeros(y.shape[0], y.shape[1], 2)
        
        # for i in range(y.shape[0]):
        #     # set y[i,:]  y[i, 0,1:3]
        #     values = y[i, 0, 1:3]
        #     new_y[i, :, 0] = values[0]
        #     new_y[i, :, 1] = values[1]
        # # for each element in the  batch replace 
        # y = new_y.to(DEVICE)
        
        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        

        y_pred = model(X)
        #print(y_pred.shape, y.shape)
        
        # [4,2000,3]
        # for first sequence draw a plotly graph of probability of first class for y and y_pred
        # if (it % 50 ==0):
        #     print(y_pred)
        #     #softmaxed = nn.functional.softmax(y_pred, dim=-1)
        #     # Create a trace for class 0 in y_pred
        #     trace0 = go.Scatter(
        #         y=y_pred[0, :, 0].detach().cpu().numpy(),
        #         mode='lines',
        #         name='y_pred_class_win'
        #     )

        #     # Create a trace for y
        #     trace_y = go.Scatter(
        #         y=y[0, :, 0].detach().cpu().numpy(),
        #         mode='lines',
        #         name='y'
        #     )

        #     # Create a Figure and add the traces for class 0
        #     fig0 = go.Figure([trace0, trace_y])

        #     # Show the figure for class 0
        #     fig0.show()
            
        
        # y_pred = y_pred.view(-1, output_dim)
        # y = y[:,:,0].view(-1, output_dim)
        
        
        
        # compute the loss
        loss = criterion(y_pred, y)
        #print(f'Loss: {loss}')
        # append the loss to the list
        losses.append(loss.item())

        # backward pass
        loss.backward()

        # update the parameters
        optimizer.step()
        # if (it*BATCH_SIZE % 200 == 0):
        #     print(f'Loss: {torch.mean(torch.tensor(losses))}')
        #     # draw graph of loss
        #     fig = px.line(x=range(len(losses)), y=losses)
        #     # Title
        #     fig.update_layout(
        #         title="Loss over time",
        #         xaxis_title="Iteration",
        #         yaxis_title="Loss",
        #         font=dict(
        #             family="Courier New, monospace",
        #             size=18,
        #             color="#7f7f7f"
        #         )
        #     )
        #     fig.show()
        it += 1
    # print the loss
    # print(f'Epoch: {epoch}, Loss: {torch.mean(torch.tensor(losses))}')
        

import numpy as np

max_game_len = X.shape[1]

accuracy_per_timestep = np.zeros(max_game_len)
added_preds = np.zeros(max_game_len)

model.eval()

X_orig = X_orig.to(DEVICE)
y_orig = y_orig.to(DEVICE)

for game_X, game_y in zip(X_orig, y_orig):
    game_X = game_X.unsqueeze(0)
    game_y = game_y.unsqueeze(0)

    game_y = game_y.squeeze().detach().cpu().numpy()
    game_y_pred = model(game_X).squeeze().detach().cpu().numpy()

    game_len = np.where((game_y[:,1:].sum(axis=-1)) == 1)[0][-1] + 1
    accuracy = (game_y_pred.argmax(axis=-1) == game_y.argmax(axis=-1))
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
# %%
