from transformer import TransformerModel

from tqdm import tqdm

import polars as pl
import torch
import torch.optim as optim
import torch.nn as nn

EPOCHS = 10
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = 'cpu'

df = pl.read_parquet("transformed_data/timeline_0.parquet")

games = []
grouped = df.group_by('matchId')
for name, group in grouped:
    group = group.drop('matchId')
    games.append(torch.from_numpy(group.to_numpy()))

# 58 per player, 48 + 1 + 9
# 227 items (?)
# 70 runes (?)
model = TransformerModel(1, 10, 5, 120, 0, 48, 22700, 1670, 70000, 50, 0, 30, 20, 30, 5)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(EPOCHS)):
    # set model to train mode
    model.train()

    # create a list to store the losses
    losses = []

    # loop over the data
    for game in games:
        X = game[:-1]
        y = game[-1]

        X = X.unsqueeze(0).to(torch.float)
        y = y.unsqueeze(0).to(torch.float)

        # send to device
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
    print(f"Epoch: {epoch}, Loss: {torch.mean(losses)}")
