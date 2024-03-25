#%%
import polars as pl
import os
from tqdm import tqdm


folder_name = "timeline_parquets_chunked"
files_list = sorted(os.listdir(folder_name))



df = pl.scan_parquet(os.path.join(folder_name, files_list[0])).collect()

cols = df.columns

playerscols = []

for i in range(1, 11):
    playercols = [c for c in cols if c.startswith(f"{i}_") ]
    # newcols.extend(sorted(playercols))
    playerscols.extend(playercols)

cols = [c for c in cols if c not in playerscols]

# print(playerscols)
# print(cols)

cols_to_one_hot = ["monsterSubType","creatorId","killType","killerId","killerTeamId",\
                    "laneType","levelUpType","monsterType","participantId","skillSlot","teamId","towerType","buildingType","type","victimId","wardType"]

df = df.to_dummies(cols_to_one_hot)

# print(list(df.select(pl.col("actualStartTime")).head(1000).unique()))

cols = df.columns

cols = [c for c in cols if c not in playerscols]

to_remove = [c for c in cols if c.endswith("_null")]
to_remove.extend(['sfg','gameId','afterId','beforeId','actualStartTime','realTimestamp','itemId','winningTeam','matchId'])

cols = [c for c in cols if c not in to_remove]

cols = ["matchId"] + cols + ['itemId'] + playerscols + ["winningTeam"]

# print(cols)

for file in tqdm(files_list):

    df = pl.scan_parquet(os.path.join(folder_name, files_list[0])).collect()

    df = df.to_dummies(cols_to_one_hot)

    df = df.select(cols)

    df = df.fill_null(0)

    print(df.columns)

    df.write_parquet(f"transformed_data/{file}",compression="zstd",compression_level=10,use_pyarrow=True)
# %%
