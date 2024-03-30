#%%
from transformer import TransformerModel

# from dotenv import load_dotenv
# load_dotenv()

from tqdm import tqdm
import polars as pl
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import plotly.graph_objects as go
import plotly.express as px
import lovely_tensors as lt
import numpy as np
import os
lt.monkey_patch()
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
BATCH_SIZE = 4

print(f'Device: {DEVICE}')

class LoLDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


#%%
# Get all files

folder_name = "transformed_data"
files_list = os.listdir(folder_name)

example_df = pl.read_parquet(os.path.join(folder_name, files_list[0]))

# for name, group in grouped:
#     group = group.drop('matchId')
#     games.append(torch.from_numpy(group.to_numpy()))


#%% Function that will take a function(game_df) -> dict (stats) and return a list of dicts

def get_stats(files_list, func):
    stats = []
    for file in files_list:
        if file.endswith(".parquet"):
            df = pl.read_parquet(os.path.join(folder_name, file))
            for name, group in tqdm(df.group_by('matchId')):
                group = group.drop('matchId')
                stats.append(func(group))
                #break
    return stats

#%%


def wining_team(game_df):
    wining_team = game_df['winningTeam'][-1]
    return {
        'winning_team': wining_team
    }


monster_types = [
    'monsterType_BARON_NASHOR',
    'monsterType_DRAGON',
    'monsterType_RIFTHERALD',]

dragon_types = ['monsterSubType_AIR_DRAGON',
    'monsterSubType_CHEMTECH_DRAGON',
    'monsterSubType_EARTH_DRAGON',
    'monsterSubType_ELDER_DRAGON',
    'monsterSubType_FIRE_DRAGON',
    'monsterSubType_HEXTECH_DRAGON',
    'monsterSubType_WATER_DRAGON']

objective_kills = [
 'type_BUILDING_KILL',
 'type_CHAMPION_KILL',]

tower_types = [
     'towerType_BASE_TURRET',
 'towerType_INNER_TURRET',
 'towerType_NEXUS_TURRET',
 'towerType_OUTER_TURRET',]

building_types = [ 'buildingType_INHIBITOR_BUILDING',
 'buildingType_TOWER_BUILDING',]

ward_objectives = [ 'type_WARD_KILL',
 'type_WARD_PLACED',]

objectives_meta_list = [monster_types, dragon_types, objective_kills, tower_types, building_types, ward_objectives]

def first_kill(game_df):
    def get_killer(objective_type):
        killer_np = game_df.filter([pl.col(objective_type)==1])[['killerId_0.0',
            'killerId_1.0',
            'killerId_10.0',
            'killerId_2.0',
            'killerId_3.0',
            'killerId_4.0',
            'killerId_5.0',
            'killerId_6.0',
            'killerId_7.0',
            'killerId_8.0',
            'killerId_9.0']].to_numpy()
        if len(killer_np) == 0:
            return None
        #print(killer_np)
        return np.where(killer_np[0] == 1)
        
    killable = [monster_types, dragon_types, objective_kills, tower_types, building_types, ['type_WARD_KILL']]
    
    first_kill_team = {
        "monster_types": {},
        "dragon_types": {},
        "objective_kills": {},
        "tower_types": {},
        "building_types": {},
        "ward_objectives": {},
    }
    
    
    for killable_class_name, killable_list in zip(first_kill_team, killable):
        for objective_type in killable_list:
            killer_id = get_killer(objective_type)
            if killer_id is None:
                continue
            
            killer_id = killer_id[0][0]
            killing_team = 100.0 if killer_id <= 5 else 200.0
            first_kill_team[killable_class_name][objective_type] = killing_team
        
        
    return {"first_kill" :first_kill_team}

def objective_counters(game_df):
    
    objective_counters = {
        "monster_types": {},
        "dragon_types": {},
        "objective_kills": {},
        "tower_types": {},
        "building_types": {},
        "ward_objectives": {},
    }
    
    for counter_name, counter in zip(objective_counters, objectives_meta_list):
        for counter_type in counter:
            objective_counters[counter_name][counter_type] = game_df.sum()[counter_type].to_list()[0]
            
    return {"objective_counters": objective_counters}
    
def game_length_and_count(game_df):
    return {
        "game_length": game_df["timestamp"].to_numpy()[-1],
        "game_count": 1
    }


def gold_advantage_as_win_predictor(game_df):
    # sum [1-5]_totalGold and [6-10]_totalGold for each row
    # if [1-5]_totalGold > [6-10]_totalGold, team 100 wins
    # else team 200 wins
    # add a column "seconds" timestamp/1000
    # wining_team = game_df['winningTeam'][-1]
    # finaly create a numpy array [60*60] with 1 if prediction is correct, 0 otherwise
    
    game_df = game_df.with_columns([
        (game_df['1_totalGold'] + game_df['2_totalGold'] + game_df['3_totalGold'] + game_df['4_totalGold'] + game_df['5_totalGold']).alias("total_gold_team_100"),
        (game_df['6_totalGold'] + game_df['7_totalGold'] + game_df['8_totalGold'] + game_df['9_totalGold'] + game_df['10_totalGold']).alias("total_gold_team_200")])


    wining_team = (game_df['winningTeam'][-1]-100.0)/100.0
    
    game_df = game_df.with_columns([game_df['timestamp'].alias("seconds")/1000.0])
    game_df = game_df.filter([pl.col("seconds") < 60*60])
    
    game_df = game_df.with_columns([((game_df['total_gold_team_100'] < game_df['total_gold_team_200']).cast(pl.Int32).alias("prediction"))])
    game_df = game_df.with_columns([((game_df['prediction'] == wining_team).cast(pl.Int32).alias("correct"))])
    
    # add column with ones
    game_df = game_df.with_columns([pl.Series("ones", [1]*game_df.height).alias("ones")])
    
    grouped_validation = game_df.group_by("seconds").agg(pl.sum("ones"))#.sort("seconds")
    grouped_prediction = game_df.group_by("seconds").agg(pl.sum("correct"))#.sort("seconds")

    correct = grouped_prediction['correct'].to_numpy()
    valid = grouped_validation['ones'].to_numpy()
    
    # now to similar thing but group not by seconds but by percentage of game length (100 buckets)
    bucket_size = game_df['timestamp'][-1]/100
    game_df = game_df.with_columns([((game_df['timestamp']/bucket_size ).cast(pl.Int32).alias("percentage"))])
 
    grouped_prediction_percentage = game_df.group_by("percentage").agg(pl.sum("correct")).sort("percentage")
    grouped_validation_percentage = game_df.group_by("percentage").agg(pl.sum("ones")).sort("percentage")
    
    correct_percentage = grouped_prediction_percentage['correct'].to_numpy()
    valid_percentage = grouped_validation_percentage['ones'].to_numpy()
    
    return {
        'correct_gold_predictions': correct,
        'valid_gold_predictions': valid,
        'correct_gold_predictions_percentage': correct_percentage,
        'valid_gold_predictions_percentage': valid_percentage
    }

game1 = None

def game_stats(game_df):
    global game1
    game1 = game_df
    stats = {}
    functions = [gold_advantage_as_win_predictor,wining_team,first_kill,objective_counters,game_length_and_count]
    for f in functions:
        stats.update(f(game_df))
    return stats
    
# %%

stats = get_stats(files_list[:1], game_stats)

#print(stats)

# %%

def post_process_stats(stats):
    output = {}
    
    # game_length and game_count
    game_length = []
    game_count = []
    for game_stats in stats:
        game_length.append(game_stats['game_length'])
        game_count.append(game_stats['game_count'])
    game_length = np.array(game_length)/60000
    output['game_length_average'] = game_length.mean()
    output['game_count'] = np.array(game_count).sum()
    output['game_length_total'] = game_length.sum()
    output['game_length'] = game_length
    
    
    # win percentage
    wining_team = []
    for game_stats in stats:
        wining_team.append(game_stats['winning_team'])
    w = np.array(wining_team)
    unique, counts = np.unique(w, return_counts=True)
    output['win_percentage'] = dict(zip(unique, counts / len(w)))
    
    # objective_counters
    # for each objective type, and sub type, count the number of times it was killed
    objective_counters = {}
    for game_stats in stats:
        for objective_type in game_stats['objective_counters']:
            if objective_type not in objective_counters:
                objective_counters[objective_type] = {}
            for sub_type in game_stats['objective_counters'][objective_type]:
                if sub_type not in objective_counters[objective_type]:
                    objective_counters[objective_type][sub_type] = 0
                objective_counters[objective_type][sub_type] += game_stats['objective_counters'][objective_type][sub_type]
                
    output['objective_counters'] = objective_counters
    
    # first_kill
    # for each objective class, and subtype, count times each team got the first kill (100 or 200)
    first_kill_count = {}
    for game_stats in stats:
        for objective_type in game_stats['first_kill']:
            if objective_type not in first_kill_count:
                first_kill_count[objective_type] = {}
            for sub_type in game_stats['first_kill'][objective_type]:
                if sub_type not in first_kill_count[objective_type]:
                    first_kill_count[objective_type][sub_type] = {100.0: 0, 200.0: 0}
                first_kill_count[objective_type][sub_type][game_stats['first_kill'][objective_type][sub_type]] += 1
    
    output['first_kill_count'] = first_kill_count
    
    # first_kill as win predictor, for each objective class, and subtype, count times each team got the first kill and won
    first_kill_win = {}
    for game_stats in stats:
        for objective_type in game_stats['first_kill']:
            if objective_type not in first_kill_win:
                first_kill_win[objective_type] = {}
            for sub_type in game_stats['first_kill'][objective_type]:
                if sub_type not in first_kill_win[objective_type]:
                    first_kill_win[objective_type][sub_type] = {100.0: 0, 200.0: 0}
                if game_stats['first_kill'][objective_type][sub_type] == game_stats['winning_team']:
                    first_kill_win[objective_type][sub_type][game_stats['first_kill'][objective_type][sub_type]] += 1
    
    output['first_kill_win'] = first_kill_win
    
    # first_kill win ratio
    first_kill_win_ratio = {}
    for objective_type in first_kill_win:
        if objective_type not in first_kill_win_ratio:
            first_kill_win_ratio[objective_type] = {}
        for sub_type in first_kill_win[objective_type]:
            if sub_type not in first_kill_win_ratio[objective_type]:
                first_kill_win_ratio[objective_type][sub_type] = {100.0: 0, 200.0: 0, 'average': 0}
            total = 0
            count = 0
            for team in first_kill_win[objective_type][sub_type]:
                first_kill_win_ratio[objective_type][sub_type][team] = first_kill_win[objective_type][sub_type][team] / first_kill_count[objective_type][sub_type][team]
                total += first_kill_win_ratio[objective_type][sub_type][team]
                count += 1
            first_kill_win_ratio[objective_type][sub_type]['average'] = total / count if count > 0 else 0
            
    output['first_kill_win_ratio'] = first_kill_win_ratio
    
    # gold advantage as win predictor
    correct_total = np.zeros(60*60)
    valid_total = np.zeros(60*60)
    
    correct_total_percentage = np.zeros(101)
    valid_total_percentage = np.zeros(101)
    
    for game_stats in stats:
        length = len(game_stats['correct_gold_predictions'])
        correct_total[0:length] += game_stats['correct_gold_predictions']
        valid_total[0:length] += game_stats['valid_gold_predictions']
        length = len(game_stats['correct_gold_predictions_percentage'])
        correct_total_percentage[0:length] += game_stats['correct_gold_predictions_percentage']
        valid_total_percentage[0:length] += game_stats['valid_gold_predictions_percentage']
    
    
    # find first 0 in valid_total
    first_zero = np.where(valid_total == 0)[0][0]
    correct_total = correct_total[:first_zero]
    valid_total = valid_total[:first_zero]
    output['gold_advantage_win_ratio'] = correct_total / valid_total
    output['valid_gold_predictions'] = valid_total
    output['correct_gold_predictions_percentage'] = correct_total_percentage / valid_total_percentage
    output['valid_gold_predictions_percentage'] = valid_total_percentage

    
    return output
# %%
post_stats = post_process_stats(stats)
#print(post_stats)
if not os.path.exists('stats'):
    os.makedirs('stats')
# %%
from plotly.subplots import make_subplots 
# draw a graph of gold_advantage_win_ratio
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=np.arange(len(post_stats['correct_gold_predictions_percentage'])), y=post_stats['correct_gold_predictions_percentage'], name='Gold Advantage Win Ratio'),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(x=np.arange(len(post_stats['valid_gold_predictions_percentage'])), y=post_stats['valid_gold_predictions_percentage'],name='Valid Gold Predictions'),
    secondary_y=True
)

fig.update_layout(
    title_text='Gold Advantage as Win Predictor',
    xaxis_title_text='Seconds',
    yaxis_title_text='Win Ratio',
    yaxis2=dict(title='Valid Gold Predictions', overlaying='y', side='right')
)

fig.show()

# %%
# Create a table for Objective Counters , columns: objective_type, sub_type, count
# polars takes dict of lists as input

# create folder stats if it does not exist


def create_objective_counters_table(post_stats):
    objective_counters = post_stats['objective_counters']
    objective_counters_dict = {"objective_type": [], "sub_type": [], "count": []}
    for objective_type in objective_counters:
        for sub_type in objective_counters[objective_type]:
            objective_counters_dict["objective_type"].append(objective_type)
            objective_counters_dict["sub_type"].append(sub_type)
            objective_counters_dict["count"].append(objective_counters[objective_type][sub_type])
    return pl.DataFrame(objective_counters_dict)



# Table for first_kill, columns: objective_type, sub_type, team_100_count, team_200_count, team_100_win_ratio, team_200_win_ratio

def create_first_kill_table(post_stats):
    first_kill_count = post_stats['first_kill_count']
    first_kill_win_ratio = post_stats['first_kill_win_ratio']
    first_kill_dict = {"objective_type": [], "sub_type": [], "team_100_count": [], "team_200_count": [], "team_100_win_ratio": [], "team_200_win_ratio": [], "average_win_ratio": []}
    for objective_type in first_kill_count:
        for sub_type in first_kill_count[objective_type]:
            first_kill_dict["objective_type"].append(objective_type)
            first_kill_dict["sub_type"].append(sub_type)
            first_kill_dict["team_100_count"].append(first_kill_count[objective_type][sub_type][100.0])
            first_kill_dict["team_200_count"].append(first_kill_count[objective_type][sub_type][200.0])
            first_kill_dict["team_100_win_ratio"].append(first_kill_win_ratio[objective_type][sub_type][100.0])
            first_kill_dict["team_200_win_ratio"].append(first_kill_win_ratio[objective_type][sub_type][200.0])
            first_kill_dict["average_win_ratio"].append(first_kill_win_ratio[objective_type][sub_type]['average'])
    return pl.DataFrame(first_kill_dict)



table_generating_functions = [create_objective_counters_table, create_first_kill_table]
table_names = ['objective_counters.csv', 'first_kill.csv']

for f, name in zip(table_generating_functions, table_names):
    df = f(post_stats)
    save_path = os.path.join('stats', name)
    df.write_csv(save_path)

# %%
# Plot histograms for game length 
fig = go.Figure()
fig.add_trace(go.Histogram(x=post_stats['game_length']))
fig.update_layout(title_text=f'Game Length Distribution (total {post_stats["game_count"]} games)', xaxis_title_text='Game Length (minutes)', yaxis_title_text='Count')
# Add mean and median lines
game_length_median = np.median(post_stats['game_length'])

fig.show()



# %%
