#%%
from transformer import TransformerModel

from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
import polars as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

EPOCHS = 10
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

games = []
grouped = df.group_by(['matchId'])
for name, group in grouped:
    group = group.drop('matchId')
    games.append(torch.from_numpy(group.to_numpy()))

games = pad_sequence(games, batch_first=True).to(torch.float)

X = games[:, :, :-1]
y = nn.functional.one_hot((games[:, :, -1] / 100.0).to(torch.long)).to(torch.float)

dataset = LoLDataset(X, y)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 58 per player, 48 + 1 + 9
# 227 items (?)
# 70 runes (?)
model = TransformerModel(3, 10, 5, 120, 0, 48, 22700, 1670, 70000, 50, 0, 30, 20, 30, 5).to(DEVICE)

# print number of parameters
print(f'Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

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

        y_pred = y_pred.view(-1, y_pred.shape[-1])
        y = y.view(-1, y.shape[-1])
        
        # compute the loss
        loss = criterion(y_pred, y)

        # append the loss to the list
        losses.append(loss.item())

        # backward pass
        loss.backward()

        # update the parameters
        optimizer.step()

    # print the loss
    print(f'Epoch: {epoch}, Loss: {torch.mean(torch.tensor(losses))}')

# %%
