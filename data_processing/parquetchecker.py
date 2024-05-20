from tqdm import tqdm
import polars as pl
import os
import torch
from torch.nn.utils.rnn import pad_sequence

folder_name = "../filtered_data"
files_list = sorted(os.listdir(folder_name))

MAX_LEN = 2928

shapes = []

for file in tqdm(files_list):

    df = pl.scan_parquet(os.path.join(folder_name, file)).collect()

    grouped = df.group_by(['matchId'])
    games = []
    for _, group in grouped:
        group = group.drop('matchId')
        games.append(torch.from_numpy(group.to_numpy()))
            
    games = pad_sequence(games, batch_first=True).to(torch.float)

    if games.shape[1] != MAX_LEN:
        padding = torch.zeros((games.shape[0], MAX_LEN - games.shape[1], games.shape[2]))
        games = torch.cat((games, padding), 1)

    shapes.append(games.shape)


for i, shape in enumerate(shapes):
    print(f"{files_list[i]}: {shape}")