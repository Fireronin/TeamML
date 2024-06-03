import json
from dataset import get_loaders
from transformer import get_model
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import plotly.graph_objects as go
import lovely_tensors as lt
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

lt.monkey_patch()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {DEVICE}')

EPOCHS = 10
EVALUATE = False
DATA_FOLDER = 'filtered_data'
GRAPHS_FOLDER = 'training_graphs'
CHECKPOINTS_FOLDER = 'checkpoints'
CHECKPOINT_FILE = 'checkpoint_5.pth'

if not os.path.exists(GRAPHS_FOLDER):
    os.makedirs(GRAPHS_FOLDER)
    
if not os.path.exists(CHECKPOINTS_FOLDER):
    os.makedirs(CHECKPOINTS_FOLDER)

with open('data_stats.json', 'r') as file:
    data_stats = json.load(file)

train_loader, test_loader = get_loaders(data_stats['max_len'], data_stats['n_games'], DATA_FOLDER, DEVICE)

model = get_model(data_stats['mean'], data_stats['std'], DEVICE)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

checkpoint_epoch = -1

if CHECKPOINT_FILE is not None:
    checkpoint = torch.load(os.path.join(CHECKPOINTS_FOLDER, CHECKPOINT_FILE))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    checkpoint_epoch = checkpoint['epoch']

for epoch in tqdm(range(EPOCHS)):
    if epoch <= checkpoint_epoch:
        continue
    
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
    if not os.path.exists(os.path.join(GRAPHS_FOLDER, 'loss')):
        os.makedirs(os.path.join(GRAPHS_FOLDER, 'loss'))
    fig.write_image(os.path.join(GRAPHS_FOLDER, 'loss', f'{epoch}.png'))

    if EVALUATE:
        model.eval()

        accuracy_per_timestep = np.zeros(data_stats['max_len'])
        added_preds = np.zeros(data_stats['max_len'])

        max_training_len = 0
        for X, y in test_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            y = y.squeeze(-1).detach().cpu().numpy()
            y_pred = model(X).squeeze(-1).detach().cpu().numpy()
            X = X.detach().cpu().numpy()

            y_pred[y_pred >= 0] = 1
            y_pred[y_pred < 0] = 0 

            accuracy = (y_pred == y)

            for game_X, game_accuracy in zip(X, accuracy): 
                nonzero = np.nonzero(game_X[:,-10])
                game_len = nonzero[-1][-1] + 1 if len(nonzero) > 0 else 0
                max_training_len = max(max_training_len, game_len)
                accuracy_per_timestep[:game_len] += game_accuracy[:game_len]
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
        if not os.path.exists(os.path.join(GRAPHS_FOLDER, 'accuracy')):
            os.makedirs(os.path.join(GRAPHS_FOLDER, 'accuracy'))
        fig.write_image(os.path.join(GRAPHS_FOLDER, 'accuracy', f'{epoch}.png'))