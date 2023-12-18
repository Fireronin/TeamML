import os
from tqdm import tqdm
import lzma
import json
import pandas as pd



FOLDER_NAME = 'raw_data/match'

# folder_name = r"C:\Users\nudy1\Downloads\raw_data\timeline"
files_list = os.listdir(FOLDER_NAME)
files_list.sort()
number_to_read = os.listdir(FOLDER_NAME).__len__()

def process_game(game: dict):
	row = {}
	row["matchId"] = game["metadata"]["matchId"]
	for p in game["info"]["participants"]:
		id = p["participantId"]
		row[f"{id}_championId"] = p["championId"]
		row[f"{id}_runeDefense"] = p["perks"]["statPerks"]["defense"]
		row[f"{id}_runeFlex"] = p["perks"]["statPerks"]["flex"]
		row[f"{id}_runeOffense"] = p["perks"]["statPerks"]["offense"]
		row[f"{id}_perk1"] = p["perks"]["styles"][0]["selections"][0]
		row[f"{id}_perk2"] = p["perks"]["styles"][0]["selections"][1]
		row[f"{id}_perk3"] = p["perks"]["styles"][0]["selections"][2]
		row[f"{id}_perk4"] = p["perks"]["styles"][0]["selections"][3]

		row[f"{id}_perk5"] = p["perks"]["styles"][1]["selections"][0]
		row[f"{id}_perk6"] = p["perks"]["styles"][1]["selections"][1]
	return pd.DataFrame(row)
dfs = []
for file in tqdm(files_list):
	if file.endswith(".xz"):
		# procesed_game = None
		with lzma.open(os.path.join(FOLDER_NAME, file), "rb") as f:
			game = json.load(f)
			processed_game = process_game(game)
			dfs.append(processed_game)


			# print (processed_game.head())
			# break
df = pd.concat(dfs)
df.to_parquet(f"parquets/match_basic.parquet",compression="zstd")
# if __name__ == '__main__':
    