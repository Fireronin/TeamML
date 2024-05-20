#%%
import polars as pl
import os
from tqdm import tqdm


folder_name = "../timeline_new"
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

import json

with open('../mapping_data/runes_data.json', 'r') as file:
    runes = json.load(file)

with open('../mapping_data/items_data.json', 'r') as file:
    items = json.load(file)

with open('../mapping_data/champs_data.json', 'r') as file:
    champs = json.load(file)

for file in tqdm(files_list):

    df = pl.scan_parquet(os.path.join(folder_name, file)).collect()

    for id in range(1,11):
        for suffix in ['runeDefense','runeFlex', 'runeOffense', 'perk1', 'perk2', 'perk3', 'perk4', 'perk5', 'perk6']:
            col = df.get_column(f"{id}_{suffix}")
            # col = col.map_elements(lambda t: runes[str(float(t))])
            col = col.cast(pl.Float32).cast(pl.String).replace(runes).cast(pl.Int32)
            df = df.with_columns(col.alias(f"{id}_{suffix}"))

        col = df.get_column(f"{id}_championId")
        # col = col.map_elements(lambda t: champs[str(float(t))])
        col = col.cast(pl.Float32).cast(pl.String).replace(champs).cast(pl.Int32)
        df = df.with_columns(col.alias(f"{id}_championId"))

    col = df.get_column(f"itemId")
    # col = col.map_elements(lambda t: items[str(t)])
    col = col.cast(pl.String).replace(items).cast(pl.Int32)
    df = df.with_columns(col.alias(f"itemId"))

    df = df.to_dummies(cols_to_one_hot)

    for col in cols:
        if col not in df.columns:
            df.insert_column(-1, pl.Series(col,[0.0] * df.height))

    df = df.select(cols)

    df = df.fill_null(0)

    # print(df.columns)

    df.write_parquet(f"../transformed_data/{file}", compression="zstd", compression_level=10, use_pyarrow=True)
# %%