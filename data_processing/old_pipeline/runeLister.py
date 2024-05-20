#%%
import json
import os
import pandas as pd
import math

df = pd.read_parquet(os.path.join("../parquets", "match_basic.parquet"))

runes = set()
champs = set()

# print(df[["1_championId","2_championId","3_championId","4_championId","5_championId","6_championId","7_championId","8_championId","9_championId","10_championId"]])

# print(df.isnull().any())

badMathes = set()

for index, row in df.iterrows():
    for id in range(1,11):
        for suffix in ['runeDefense','runeFlex', 'runeOffense', 'perk1', 'perk2', 'perk3', 'perk4', 'perk5', 'perk6']:
            rune = row[f"{id}_{suffix}"]

            if type(rune) == int:
                runes.add(float(rune))

            elif math.isnan(rune):
                badMathes.add(row[f"matchId"])
            else:
                runes.add(rune)

        if type(row[f"{id}_championId"]) == int:
            champs.add(float(row[f"{id}_championId"]))
        elif type(row[f"{id}_championId"]) == float and math.isnan(row[f"{id}_championId"]):

            badMathes.add(row[f"matchId"])
        else:
            champs.add(row[f"{id}_championId"])

    # print(row)

print(champs)

print(runes)

rune_dict = {k: v for k, v in zip(list(sorted(runes)), range(1, len(runes) + 1))}
runes_data = json.dumps(rune_dict)

champs_dict = {k: v for k, v in zip(list(sorted(champs)), range(1, len(champs) + 1))}
champs_data = json.dumps(champs_dict)

# Write JSON data to a file
with open('../mapping_data/runes_data.json', 'w') as file:
    file.write(runes_data)

with open('../mapping_data/champs_data.json', 'w') as file:
    file.write(champs_data)

print(badMathes)
# %%
