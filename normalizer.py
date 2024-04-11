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
import os
import gc
import json

lt.monkey_patch()

EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
BATCH_SIZE = 8

DATA_FOLDER = 'transformed_data'
GRAPHS_FOLDER = 'training_graphs'
CHECKPOINTS_FOLDER = 'checkpoints'



files_list = os.listdir(DATA_FOLDER)

# dfs = []

n_games = 0

sum = {}
count = 0

max_len = 0

for file in files_list:
    df = pl.read_parquet(os.path.join(DATA_FOLDER, file))
    # dfs.append(df)

    count += df.shape[0]

    for col in df.columns:

        if col == 'matchId':
            continue
        if col not in sum:
            sum[col] = 0
        sum[col] += df[col].sum()

    grouped = df.group_by(['matchId'])

    max_len = max(max_len, grouped.count()["count"].max())

    n_games += grouped.all().shape[0]

    del df

mean = {col: sum[col] / count for col in sum}


# print(sum)

squarediffsum = {}

for file in files_list:
    df = pl.read_parquet(os.path.join(DATA_FOLDER, file))
    # dfs.append(df)

    # count += df.shape[0]

    for col in df.columns:
        if col == 'matchId':
            continue
        if col not in squarediffsum:
            squarediffsum[col] = 0
        squarediffsum[col] += ((df[col] - mean[col]) ** 2).sum()


std = {col: (squarediffsum[col] / count) ** 0.5 for col in squarediffsum}


mean_list = []
std_list = []

df = pl.read_parquet(os.path.join(DATA_FOLDER, files_list[0]))
for col in df.columns:
    if col == 'matchId':
        continue
    mean_list.append(mean[col])
    std_list.append(std[col] if std[col] != 0 else 1)


stats = {
    'max_len': max_len,
    'n_games': n_games,
    'mean': mean_list,
    'std': std_list
}

with open('data_stats.json', 'w') as file:
    json.dump(stats, file)
