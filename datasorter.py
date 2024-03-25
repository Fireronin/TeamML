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

with open('runes_data.json', 'r') as file:
    runes = json.load(file)

with open('items_data.json', 'r') as file:
    items = json.load(file)

with open('champs_data.json', 'r') as file:
    champs = json.load(file)

for file in tqdm(files_list[:2]):

    df = pl.scan_parquet(os.path.join(folder_name, file)).collect()

    for id in range(1,11):
        for suffix in ['runeDefense','runeFlex', 'runeOffense', 'perk1', 'perk2', 'perk3', 'perk4', 'perk5', 'perk6']:
            # rune = row[f"{id}_{suffix}"]
            col = df.get_column(f"{id}_{suffix}")
            col = col.map_elements(lambda t: runes[t])  
            df.replace(f"{id}_{suffix}", col)
            # for row in col.iter_rows():

        col = df.get_column(f"{id}_championId")
        col = col.map_elements(lambda t: champs[t])  
        df.replace(f"{id}_championId", col)

    col = df.get_column(f"itemId")
    col = col.map_elements(lambda t: items[t])  
    df.replace(f"itemId", col)

    df = df.to_dummies(cols_to_one_hot)

    # necessary_columns = ['wardType_TEEMO_MUSHROOM']

    for col in cols:
        if col not in df.columns:
            df.insert_column(-1, pl.Series(col,[0.0] * df.height))

    df = df.select(cols)

    df = df.fill_null(0)

    print(df.columns)

    df.write_parquet(f"transformed_data/{file}", compression="zstd", compression_level=10, use_pyarrow=True)