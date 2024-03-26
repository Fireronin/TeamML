#%%
import json
import os
# import polars as pl
from tqdm import tqdm
import numpy as np
import lzma
import pandas as pd
import concurrent.futures
import more_itertools
import itertools
from multiprocessing import Pool
# from numba import jit
# from numba.experimental import jitclass
# itterate over all files in timeline folder
# those are .json.xz files
# decompress them and read them as json in timelineDecompressed folder

#%%
folder_name = "raw_data/timeline"
files_list = os.listdir(folder_name)
files_list.sort()
number_to_read = os.listdir(folder_name).__len__()

# read first n files
files_list = files_list[:number_to_read]

		
	
# %%
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Frame:
	timestamp: int
	participantFrames: dict
	events: List[dict]


@dataclass
class GameTimeline:
	matchId: str
	frames: List[Frame]
	participats: List[Tuple]
	

games = []

#%%
def flatten_dict(d, parent_key='', sep='_'):
	items = []
	for k, v in d.items():
		new_key = parent_key + sep + k if parent_key else k
		if isinstance(v, dict):
			items.extend(flatten_dict(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)

def flatten_event(d, parent_key='', sep='_'):
	items = []
	if "assistingParticipantIds" in d:
		# one hot encode assistingParticipantIds
		for i in range(1,11):
			if i in d["assistingParticipantIds"]:
				d[f"assistingParticipantIds_{i}"] = 1
			else:
				d[f"assistingParticipantIds_{i}"] = 0
		del d["assistingParticipantIds"]
	for k, v in d.items():
		new_key = parent_key + sep + k if parent_key else k
		# if key assistingParticipantIds then it is a list list 
		# convert it to one hot encoding 1-10
		if k == "assistingParticipantIds":
			continue
		if k == "victimDamageDealt" or k == "victimDamageReceived":
			continue
		if isinstance(v,list):
			raise Exception(f"List found in event {k}")
		if isinstance(v, dict):
			items.extend(flatten_dict(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)

def process_frame(frame: dict) -> Frame:
	# frame contains timestamp, participantFrames, events
	# flatten participantFrames using _ to join keys keys of nested dicts
	participantFrames = flatten_dict(frame["participantFrames"])
	events = [flatten_event(event) for event in frame["events"]]
	return Frame(frame["timestamp"], participantFrames, events)    

def process_game(game: dict) -> GameTimeline:
	g_timeline = GameTimeline(game["metadata"]["matchId"], [], [])
	frames = []
	for frame in game["info"]["frames"]:
		frames.append(process_frame(frame))
	g_timeline.frames = frames
	participants = []
	for participant in game["info"]["participants"]:
		participants.append((participant["participantId"], participant["puuid"]))
	g_timeline.participants = participants
	return g_timeline

def process_keys(gameTimeline: GameTimeline):
	event_keys = set()
	participantFrame_keys = set()
	for frame in gameTimeline.frames:
		for event in frame.events:
			event_keys.update(event.keys())
		participantFrame_keys.update(frame.participantFrames.keys())
	return event_keys, participantFrame_keys


def game_to_df(game: GameTimeline):
	rows = []
	for frame in game.frames:
		for event in frame.events:
			# create a dict with all columns and values from event and participantFrame
			# add gameId and timestamp
			row = {}
			row["matchId"] = game.matchId
			row["timestamp"] = frame.timestamp
			# add all event keys fill with null if not present
			for key in event_keys:
				if key in event:
					row[key] = event[key]
				else:
					row[key] = None
			# add all participantFrame keys fill with null if not present
			for key in participantFrame_keys:
				if key in frame.participantFrames:
					row[key] = frame.participantFrames[key]
				else:
					row[key] = None
			rows.append(row)
	return pd.DataFrame(rows)

#%%
event_keys, participantFrame_keys = None, None
for file in tqdm(files_list):
	if file.endswith(".xz"):
		procesed_game = None
		with lzma.open(os.path.join(folder_name, file), "rb") as f:
			game = json.load(f)
			procesed_game = process_game(game)
			event_keys, participantFrame_keys = process_keys(procesed_game)
			break
#%%
# read decompressed files oldf
# dfs = []
# for file in tqdm(files_list):
# 	if file.endswith(".xz"):
# 		procesed_game = None
# 		with lzma.open(os.path.join(folder_name, file), "rb") as f:
# 			game = json.load(f)
# 			procesed_game = process_game(game)
# 			dfs.append(game_to_df(procesed_game))
# 	# every 1000 games save to parquet
# 	if len(dfs) >= 100:
# 		df = pd.concat(dfs)
# 		df.to_parquet(f"timeline_{file}.parquet",compression="zstd")
# 		dfs = []

# %%


def process_file(file):
	if file.endswith(".xz"):
		procesed_game = None
		with lzma.open(os.path.join(folder_name, file), "rb") as f:
			game = json.load(f)
			procesed_game = process_game(game)
			return game_to_df(procesed_game)

dfs = []
k = 100
chunk_count = len(files_list) // k	
ct = 0

for file_chunk in tqdm(more_itertools.chunked(files_list[ct*k:], k)):
    dfs = map(process_file, file_chunk)
    df = pd.concat(dfs)
    df.to_parquet(f"timeline_parquets/timeline_{ct}.parquet",compression="zstd")
    dfs = []
    ct+=1



# %%
