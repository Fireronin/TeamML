import polars as pl
import os
from tqdm import tqdm


folder_name = "timeline_new"
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
print(cols)

df = df.to_dummies(["monsterSubType","creatorId","killType","killerId","killerTeamId",\
                    "laneType","levelUpType","monsterType","participantId","skillSlot","teamId","towerType","type","victimId","wardType"])

# print(list(df.select(pl.col("actualStartTime")).head(1000).unique()))

cols = df.columns

cols = [c for c in cols if c not in playerscols]

to_remove = [c for c in cols if c.endswith("_null")]
to_remove.extend(['sfg','gameId','afterId','beforeId','actualStartTime','realTimestamp','itemId','winningTeam'])

cols = [c for c in cols if c not in to_remove]

cols = cols + ['itemId'] + playerscols + ["winningTeam"]

# print(cols)

for file in tqdm(files_list[:1]):

    df = pl.scan_parquet(os.path.join(folder_name, files_list[0])).collect()

    df = df.to_dummies(["monsterSubType","creatorId","killType","killerId","killerTeamId",\
                    "laneType","levelUpType","monsterType","participantId","skillSlot","teamId","towerType","type","victimId","wardType"])

    df = df.select(cols)

    df.fill_null(0)

    df.write_parquet(f"transformed_data/{file}",compression="zstd",compression_level=10,use_pyarrow=True)