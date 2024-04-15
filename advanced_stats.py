#%%

from tqdm import tqdm
import polars as pl
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
from plotly.subplots import make_subplots 
#%%
# Get all files

folder_name = "transformed_data"
files_list = os.listdir(folder_name)

example_df = pl.read_parquet(os.path.join(folder_name, files_list[0]))

# Create folder for new files
if not os.path.exists('stats'):
    os.makedirs('stats')

#%% 
game1 = None # debug

def get_stats(files_list, tuple_list,early_stop=-1):
    global game1
    states = {}
    for t in tuple_list:
        states[t[0]] = t[4]
    
    it = 0
    
    for file in files_list:
        if file.endswith(".parquet"):
            df = pl.read_parquet(os.path.join(folder_name, file))
            for name, group in tqdm(df.group_by('matchId')):
                if game1 is None:
                    game1 = group
                group = group.drop('matchId')
                for t in tuple_list:
                    stats = t[1](group)
                    states[t[0]] = t[2](stats, states[t[0]])
                if it>=early_stop and early_stop != -1:
                    return states
                it+=1
            
    return states

def post_process_stats(states, tuple_list):
    pps = {}    
    for t in tuple_list:
        pps.update(t[3](states[t[0]]))
    
    return pps

#%%

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

objectives_meta_names = ['monster_types', 'dragon_types', 'objective_kills', 'tower_types', 'building_types', 'ward_objectives']


def game_length_and_count(game_df):
    return {
        "game_length": game_df["timestamp"].to_numpy()[-1],
        "game_count": 1
    }

def game_length_and_count_pre_merge(new_stats, state=None):
    if state is None:
        state = {
            "game_length": 0.0,
            "game_count": 0,
            "game_length_buckets": np.zeros(60*60)
        }
    state["game_length"] += new_stats["game_length"]
    state["game_count"] += new_stats["game_count"]
    state["game_length_buckets"][(new_stats["game_length"]/1000).astype(int)] += 1
    return state

def game_length_and_count_post_process(state):
    average_game_length = state["game_length"] / state["game_count"]
    
    post_stats = {
        "game_length_average": average_game_length,
        "game_count": state["game_count"],
        "game_length_total": state["game_length"],
        "game_length": state["game_length_buckets"],
    }
    
    game_lengths_per_minute = post_stats['game_length']
    # combine every 60 buckets into one
    game_lengths_per_minute = np.array([np.sum(game_lengths_per_minute[i:i+60]) for i in range(0, len(game_lengths_per_minute), 60)])
    
    # draw a graph of game length distribution
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=np.arange(len(game_lengths_per_minute)), y=game_lengths_per_minute, name='Game Length Distribution'))
    fig.update_layout(title='Game Length Distribution', xaxis_title='Minutes', yaxis_title='Game Count')    
    fig.show()
    # save the graph
    #fig.write_image(os.path.join('stats', 'game_length_distribution.png'))
    return post_stats

game_length_and_count_tuple = ("game_length_and_count",game_length_and_count, game_length_and_count_pre_merge, game_length_and_count_post_process, None)


def wining_team(game_df):
    wining_team = game_df['winningTeam'][-1]
    return {
        'winning_team': wining_team
    }

def winning_team_pre_merge(new_stats, state=None):
    if state is None:
        state = {'win_percentage': {0.0:0,100.0: 0, 200.0: 0}}
    state['win_percentage'][new_stats['winning_team']] += 1
    return state

def winning_team_post_process(state):
    return state

winning_team_tuple = ("winning_team",wining_team, winning_team_pre_merge, winning_team_post_process, None)


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
    
    
    for killable_class_name, killable_list in zip(objectives_meta_names, killable):
        for objective_type in killable_list:
            killer_id = get_killer(objective_type)
            if killer_id is None:
                continue
            
            killer_id = killer_id[0][0]
            killing_team = 100.0 if killer_id <= 5 else 200.0
            first_kill_team[killable_class_name][objective_type] = killing_team
        
        
    return {
        "first_kill" :first_kill_team,
        "winning_team": game_df['winningTeam'][-1]
    }

def first_kill_pre_merge(new_stats, state=None):
    if state is None:
        state = {
            "first_kill_count": {},
            "first_kill_win": {},
        }
        for i,objective  in enumerate(objectives_meta_names):
            state['first_kill_count'][objective] = {}
            state['first_kill_win'][objective] = {}
            for sub_type in objectives_meta_list[i]:
                state['first_kill_count'][objective][sub_type] = {100.0: 0, 200.0: 0}
                state['first_kill_win'][objective][sub_type] = {100.0: 0, 200.0: 0}
    
    for objective in new_stats['first_kill']:
        for sub_type in new_stats['first_kill'][objective]:
            state['first_kill_count'][objective][sub_type][new_stats['first_kill'][objective][sub_type]] += 1
            if new_stats['first_kill'][objective][sub_type] == new_stats['winning_team']:
                state['first_kill_win'][objective][sub_type][new_stats['first_kill'][objective][sub_type]] += 1
    
    return state
    
def first_kill_post_process(state):
    first_kill_win_ratio = {}
    
    for objective in state['first_kill_win']:
        first_kill_win_ratio[objective] = {}
        for sub_type in state['first_kill_win'][objective]:
            first_kill_win_ratio[objective][sub_type] = {}
            for team in state['first_kill_win'][objective][sub_type].keys():
                if state['first_kill_count'][objective][sub_type][team] == 0:
                    first_kill_win_ratio[objective][sub_type][team] = 0
                else:
                    first_kill_win_ratio[objective][sub_type][team] = state['first_kill_win'][objective][sub_type][team] / state['first_kill_count'][objective][sub_type][team]
            first_kill_win_ratio[objective][sub_type]['average'] = np.mean(list(first_kill_win_ratio[objective][sub_type].values()))
    
    post_stats = {
        "first_kill_win_ratio": first_kill_win_ratio,
        "first_kill_count": state['first_kill_count'],
        "first_kill_win": state['first_kill_win']
    }
       
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
        
    df = create_first_kill_table(post_stats)
    save_path = os.path.join('stats', 'first_kill.csv')
    df.write_csv(save_path)
        
    return post_stats
        
first_kill_tuple =  ("first_kill", first_kill, first_kill_pre_merge, first_kill_post_process, None)


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

def objective_counters_pre_merge(new_stats, state=None):
    if state is None:
        state = {}
        for counter_name, counter in zip(objectives_meta_names, objectives_meta_list):
            state[counter_name] = {}
            for counter_type in counter:
                state[counter_name][counter_type] = 0
    
    for counter_name in new_stats['objective_counters']:
        for counter_type in new_stats['objective_counters'][counter_name]:
            state[counter_name][counter_type] += new_stats['objective_counters'][counter_name][counter_type]
    
    return state

def objective_counters_post_process(state):
    
    def create_objective_counters_table(post_stats):
        objective_counters = post_stats['objective_counters']
        objective_counters_dict = {"objective_type": [], "sub_type": [], "count": []}
        for objective_type in objective_counters:
            for sub_type in objective_counters[objective_type]:
                objective_counters_dict["objective_type"].append(objective_type)
                objective_counters_dict["sub_type"].append(sub_type)
                objective_counters_dict["count"].append(objective_counters[objective_type][sub_type])
        return pl.DataFrame(objective_counters_dict)

    post_stats = {
        "objective_counters": state
    }
    
    df = create_objective_counters_table(post_stats)
    save_path = os.path.join('stats', 'objective_counters.csv')
    df.write_csv(save_path)
    
    return post_stats

objective_counters_tuple = ("objective_counters", objective_counters, objective_counters_pre_merge, objective_counters_post_process, None)

def calculate_total_gold(game_df):
    game_df = game_df.with_columns([
        (game_df['1_totalGold'] + game_df['2_totalGold'] + game_df['3_totalGold'] + game_df['4_totalGold'] + game_df['5_totalGold']).alias("total_gold_team_100"),
        (game_df['6_totalGold'] + game_df['7_totalGold'] + game_df['8_totalGold'] + game_df['9_totalGold'] + game_df['10_totalGold']).alias("total_gold_team_200")])
    return game_df

def calculate_seconds(game_df):
    game_df = game_df.with_columns([game_df['timestamp'].alias("seconds")/1000.0])
    game_df = game_df.filter([pl.col("seconds") < 60*60])
    return game_df

def calculate_prediction(game_df, wining_team):
    game_df = game_df.with_columns([((game_df['total_gold_team_100'] < game_df['total_gold_team_200']).cast(pl.Int32).alias("prediction"))])
    game_df = game_df.with_columns([((game_df['prediction'] == wining_team).cast(pl.Int32).alias("correct"))])
    return game_df

def gold_advantage_as_win_predictor(game_df):
    game_df = calculate_total_gold(game_df)
    wining_team = (game_df['winningTeam'][-1]-100.0)/100.0
    game_df = calculate_seconds(game_df)
    game_df = calculate_prediction(game_df, wining_team)
    
    # seconds
    grouped_by_seconds = game_df[["correct","seconds"]].group_by("seconds",maintain_order=True)
    correct_seconds = grouped_by_seconds.sum()["correct"].to_numpy()
    valid_seconds = grouped_by_seconds.count()["count"].to_numpy()
    
    # percentage
    bucket_size = game_df['timestamp'][-1]/100 
    buckets_cuts = np.arange(0, 99)*bucket_size
    labels = [str(i) for i in range(100)]
    game_df = game_df.with_columns([game_df["timestamp"].cut(breaks=buckets_cuts,labels=labels).alias("percentage")])
    grouped_by_percentage =  game_df[["correct","percentage"]].group_by("percentage",maintain_order=True)
    percentage_df  = grouped_by_percentage.sum().with_columns([grouped_by_percentage.count()["count"]])
    to_add_columns = {
        "percentage": [],
        "correct": [],
        "count": [],
    }
    for i in range(100):
        if percentage_df.filter([pl.col("percentage") == str(i)]).height == 0:
            to_add_columns["correct"].append(0)
            to_add_columns["count"].append(0)
            to_add_columns["percentage"].append(str(i))
    if(len(to_add_columns["correct"]) > 0):
        to_add_df = pl.DataFrame(to_add_columns,{ "percentage": pl.Categorical,"correct": pl.Int32, "count": pl.UInt32})
        percentage_df = pl.concat([percentage_df,to_add_df],how="vertical_relaxed")
    
    percentage_df = percentage_df.sort("percentage")
    correct_percentage = percentage_df["correct"].to_numpy()
    valid_percentage = percentage_df["count"].to_numpy()
    
    # raw stamps
    correct_raw = game_df["correct"].to_numpy()
    valid_raw = np.ones(len(correct_raw))
    
    correct_raw = np.concatenate([correct_raw, np.zeros(10000-len(correct_raw))])
    valid_raw = np.concatenate([valid_raw, np.zeros(10000-len(valid_raw))])
    
    
    return {
        'gb_correct_per_second': correct_seconds,
        'gb_valid_per_second': valid_seconds,
        'gb_correct_per_percent': correct_percentage,
        'gb_valid_per_percent': valid_percentage,
        'gb_correct_raw': correct_raw,
        'gb_valid_raw': valid_raw,
    }

def gold_advantage_as_win_predictor_pre_merge(new_stats, state=None):
    if state is None:
        state = {
            'gb_correct_per_second': np.zeros(60*60),
            'gb_valid_per_second': np.zeros(60*60),
            'gb_correct_per_percent': np.zeros(100),
            'gb_valid_per_percent': np.zeros(100),
            'gb_correct_raw': np.zeros(10000),
            'gb_valid_raw': np.zeros(10000),
        }
    seconds_length = len(new_stats['gb_correct_per_second'])
    max_cut = min(len(state['gb_correct_per_second']), seconds_length)
    state['gb_correct_per_second'][:max_cut] += new_stats['gb_correct_per_second'][:max_cut]
    state['gb_valid_per_second'][:max_cut] += new_stats['gb_valid_per_second'][:max_cut]
    state['gb_correct_per_percent'] += new_stats['gb_correct_per_percent']
    state['gb_valid_per_percent'] += new_stats['gb_valid_per_percent']
    state['gb_correct_raw'] += new_stats['gb_correct_raw']
    state['gb_valid_raw'] += new_stats['gb_valid_raw']
    return state

def gold_advantage_as_win_predictor_post_process(state):
    first_zero = np.where(state['gb_valid_per_second'] == 0)[0][0]
    state['gb_correct_per_second'] = state['gb_correct_per_second']#[:first_zero]
    state['gb_valid_per_second'] = state['gb_valid_per_second']#[:first_zero]
    state['gb_correct_per_second'] = state['gb_correct_per_second'] / state['gb_valid_per_second']
    state['gb_valid_per_second'] = state['gb_valid_per_second']
    state['gb_correct_per_percent'] = state['gb_correct_per_percent'] / state['gb_valid_per_percent']
    state['gb_valid_per_percent'] = state['gb_valid_per_percent']
    
    first_zero = np.where(state['gb_valid_raw'] == 0)[0][0]
    state['gb_correct_raw'] = state['gb_correct_raw'][:first_zero]
    state['gb_valid_raw'] = state['gb_valid_raw'][:first_zero]
    state['gb_correct_raw'] = state['gb_correct_raw'] / state['gb_valid_raw']
    state['gb_valid_raw'] = state['gb_valid_raw']
    
    
    pairs_of_columns = [
        ('gb_correct_per_second', 'gb_valid_per_second'),
        ('gb_correct_per_percent', 'gb_valid_per_percent'),
        ('gb_correct_raw', 'gb_valid_raw')]
    
    titles = [
        'Gold Advantage Baseline accuracy per second',
        'Gold Advantage Baseline accuracy per percentage or match time',
        'Gold Advantage Baseline accuracy per nth timestamp']
    
    x_axis_titles = [
        'Seconds',
        'Percentage',
        'Timestamp']
    
    for pair in pairs_of_columns:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=np.arange(len(state[pair[0]])), y=state[pair[0]], name='Correct %'),secondary_y=False)
        fig.add_trace(go.Scatter(x=np.arange(len(state[pair[1]])), y=state[pair[1]], name='Number of Samples'),secondary_y=True)
        fig.update_layout(title=titles[pairs_of_columns.index(pair)], 
                          xaxis_title=x_axis_titles[pairs_of_columns.index(pair)], 
                          yaxis_title='Win Ratio',
                          yaxis2=dict(title='Valid Gold Predictions', overlaying='y', side='right'))
        fig.show()
    
    return state

gold_advantage_as_win_predictor_tuple = ("gold_advantage_as_win_predictor", gold_advantage_as_win_predictor, gold_advantage_as_win_predictor_pre_merge, gold_advantage_as_win_predictor_post_process, None)

#compute mean and std of each column (combined for all games)

def mean_of_all_games(df):
    counts = []
    sums = []
    for col in df.columns:
        if col == 'matchId':
            continue
        counts.append(df[col].len())
        sums.append(df[col].sum())
    return {
        "counts": counts,
        "sums": sums
    }
    
def mean_of_all_games_pre_merge(new_stats, state=None):
    if state is None:
        state = {
            "counts": np.zeros(len(new_stats["counts"])),
            "sums": np.zeros(len(new_stats["sums"]))
        }
    state["counts"] += new_stats["counts"]
    state["sums"] += new_stats["sums"]
    return state

def mean_of_all_games_post_process(state):
    means = state["sums"] / state["counts"]
    
    means_dict = {}
    i = 0
    for col in game1.columns:
        if col == 'matchId':
            continue
        means_dict[col] = means[i]
        i+=1
        
    means_dict['matchId'] = None
    means_df = pl.DataFrame(means_dict)
    save_path = os.path.join('stats', 'means.csv')
    means_df.write_csv(save_path)
    
    global means_global
    means_global = means_dict
    return {
        "means": means
    }
    
mean_of_all_games_tuple = ("mean_of_all_games", mean_of_all_games, mean_of_all_games_pre_merge, mean_of_all_games_post_process, None)

means_global = [] 


# compute std using means_global as mean of each column

def std_of_all_games(df):
    counts = []
    sums = []
    for col in df.columns:
        if col == 'matchId':
            continue
        counts.append(df[col].len())
        sums.append(((df[col] - means_global[col])**2).sum() )
    return {
        "counts": counts,
        "sums": sums
    }
    
def std_of_all_games_pre_merge(new_stats, state=None):
    if state is None:
        state = {
            "counts": np.zeros(len(new_stats["counts"])),
            "sums": np.zeros(len(new_stats["sums"]))
        }
    state["counts"] += new_stats["counts"]
    state["sums"] += new_stats["sums"]
    return state

def std_of_all_games_post_process(state):
    stds = np.sqrt(state["sums"] / state["counts"])
    
    stds_dict = {}
    i = 0
    for col in game1.columns:
        if col == 'matchId':
            continue
        stds_dict[col] = stds[i]
        i+=1
        
    stds_dict['matchId'] = None
    stds_df = pl.DataFrame(stds_dict)
    save_path = os.path.join('stats', 'stds.csv')
    stds_df.write_csv(save_path)
    
    return {
        "stds": stds
    }

std_of_all_games_tuple = ("std_of_all_games", std_of_all_games, std_of_all_games_pre_merge, std_of_all_games_post_process, None)


# %%
# [game_length_and_count_tuple, winning_team_tuple, first_kill_tuple, objective_counters_tuple,gold_advantage_as_win_predictor_tuple]
tuple_list = [game_length_and_count_tuple, winning_team_tuple, first_kill_tuple, objective_counters_tuple,gold_advantage_as_win_predictor_tuple]

# stats =  get_stats(files_list[:100000], tuple_list,early_stop=-1)

#%%

# post_stats = post_process_stats(stats,tuple_list)


# %%

# std and mean
stop_idx = -1
mean_stats = get_stats(files_list[:100000], [mean_of_all_games_tuple],early_stop=stop_idx)
post_mean_stats = post_process_stats(mean_stats,[mean_of_all_games_tuple])
#%%
std_stats = get_stats(files_list[:100000], [std_of_all_games_tuple],early_stop=stop_idx)
post_std_stats = post_process_stats(std_stats,[std_of_all_games_tuple])


# %%
