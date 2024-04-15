#%%
import os
from tqdm import tqdm
import lzma
import json
import pandas as pd
import math

FOLDER_NAME = '../raw_data/match'

# folder_name = r"C:\Users\nudy1\Downloads\raw_data\timeline"
files_list = os.listdir(FOLDER_NAME)
files_list.sort()
number_to_read = os.listdir(FOLDER_NAME).__len__()

def process_game(game: dict) -> pd.DataFrame:
	row = {}
	row["matchId"] = game["metadata"]["matchId"]
	row["sfg"] = 4.0
	for p in game["info"]["participants"]:
		id = p["participantId"]
		row[f"{id}_championId"] = p["championId"]

		if math.isnan(p["championId"]) or p["championId"] is None:
			print(row["matchId"])
			print(p)
			
		row[f"{id}_runeDefense"] = p["perks"]["statPerks"]["defense"]
		row[f"{id}_runeFlex"] = p["perks"]["statPerks"]["flex"]
		row[f"{id}_runeOffense"] = p["perks"]["statPerks"]["offense"]
		row[f"{id}_perk1"] = p["perks"]["styles"][0]["selections"][0]["perk"]
		row[f"{id}_perk2"] = p["perks"]["styles"][0]["selections"][1]["perk"]
		row[f"{id}_perk3"] = p["perks"]["styles"][0]["selections"][2]["perk"]
		row[f"{id}_perk4"] = p["perks"]["styles"][0]["selections"][3]["perk"]

		row[f"{id}_perk5"] = p["perks"]["styles"][1]["selections"][0]["perk"]
		row[f"{id}_perk6"] = p["perks"]["styles"][1]["selections"][1]["perk"]
	return pd.DataFrame([row])

dfs : list[pd.DataFrame] = []

for file in tqdm(files_list):
	if file.endswith(".xz"):
		# procesed_game = None
		with lzma.open(os.path.join(FOLDER_NAME, file), "rb") as f:
			game = json.load(f)
			processed_game = process_game(game)
			dfs.append(processed_game)


df = pd.concat(dfs)

df.to_parquet(f"../parquets/match_basic.parquet",compression="zstd")

#%% Rune lister

runes = set()
champs = set()
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
