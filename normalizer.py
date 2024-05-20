import polars as pl
import os
import json

DATA_FOLDER = 'filtered_data'

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
