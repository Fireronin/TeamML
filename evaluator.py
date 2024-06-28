from tqdm import tqdm
import numpy as np
import os 
import json
from dataset import get_loaders
from transformer import get_model
import torch
import plotly.graph_objects as go
import polars as pl

from dotenv import load_dotenv
load_dotenv()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {DEVICE}')

DATA_FOLDER = 'filtered_data'
GRAPHS_FOLDER = 'evaluation_graphs'
CHECKPOINTS_FOLDER = 'checkpoints'
CHECKPOINT_FILE = 'checkpoint_4.pth'

with open('data_stats.json', 'r') as file:
    data_stats = json.load(file)

model = get_model(data_stats['mean'], data_stats['std'], DEVICE, data_stats['max_len'])

train_loader, test_loader = get_loaders(data_stats['max_len'], data_stats['n_games'], DATA_FOLDER, DEVICE, calculate_timestamps=True)

checkpoint = torch.load(os.path.join(CHECKPOINTS_FOLDER, CHECKPOINT_FILE))
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']

model.eval()

accuracy_per_percent = np.zeros(101)
percentage_samples = np.zeros(101)

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

    for game_accuracy, timestamps in zip(accuracy, t):

        for acc, timestamp in zip(game_accuracy, timestamps):
            accuracy_per_percent[timestamp] += acc
            percentage_samples[timestamp] += 1


accuracy_per_percent = accuracy_per_percent / percentage_samples

trace0 = go.Scatter(
    y=accuracy_per_percent,
    mode='lines',
    name='Accuracy'
)

fig = go.Figure(data=[trace0])
if not os.path.exists(GRAPHS_FOLDER):
    os.makedirs(GRAPHS_FOLDER)
fig.write_image(os.path.join(GRAPHS_FOLDER, f'{epoch}.png'))


df = pl.DataFrame(accuracy_per_percent)
save_path = os.path.join('stats', '{epoch}_accuracy_per_percent.csv')
df.write_csv(save_path)