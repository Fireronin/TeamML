#%%
import xgboost as xgb
import sklearn.metrics
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm
import polars as pl
import os 
import json
from transformer import TransformerModel
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression

BATCH_SIZE = 150000
ITERATIONS = 1
model = None
DEVICE = 'cpu' #"torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_FOLDER = 'filtered_data/filtered_data'
GRAPHS_FOLDER = 'training_graphs'
CHECKPOINTS_FOLDER = 'checkpoints'

if not os.path.exists(GRAPHS_FOLDER):
    os.makedirs(GRAPHS_FOLDER)
    
if not os.path.exists(CHECKPOINTS_FOLDER):
    os.makedirs(CHECKPOINTS_FOLDER)


class LoLDatasetCache(Dataset):
    def __init__(self, max_len, n_games, k_samples):
        self.max_len = max_len
        self.n_games = n_games
        self.k_samples = k_samples
        self.cached_data = None
        self.cached_targets = None
        self.cached_file_number = -1
        self.cache_size = -1
        self.cached_timestamps = None
    
    def __len__(self):
        return self.n_games
    
    def __getitem__(self, idx):
        file_number = int(idx // 1000)
        if self.cached_file_number != file_number:
            file_name =  f'timeline_{file_number}.parquet'
            df = pl.read_parquet(os.path.join(DATA_FOLDER,file_name))

            grouped = df.group_by(['matchId'])
            games = []
            timestamps_per_game = []
            for _, group in grouped:
                group = group.drop('matchId')
                timestamps = group['timestamp'].to_numpy()
                tensor_group = torch.from_numpy(group.to_numpy())
                max_time = timestamps[-1]
                timestamps = (timestamps / max_time).astype(np.float32)
                # multiply by 100 to get the percentage then int
                timestamps = (timestamps * 100).astype(np.int32)
                if tensor_group.shape[0] > self.k_samples:
                    sample_indices = random.sample(range(tensor_group.shape[0]), self.k_samples)
                    # take first 10 samples
                    #sample_indices = [0]

                    
                    tensor_group = tensor_group[sample_indices]
                    
                    timestamps = timestamps[sample_indices]
                    
                if tensor_group.shape[0] < self.k_samples:
                    continue
                timestamps_per_game.append(torch.from_numpy(timestamps))
                games.append(tensor_group)
            
            games = torch.stack(games).to(torch.float)

            timestamps_per_game = torch.stack(timestamps_per_game).to(DEVICE)

            games[:, :, -1] = games[:, 0, -1].unsqueeze(-1).to(DEVICE)
            X = games[:, :, :-1]
            y = (games[:, :, -1] / 100.0 - 1).unsqueeze(-1)

            self.cached_data = X
            self.cached_targets = y
            self.cached_file_number = file_number
            self.cache_size = games.shape[0]
            self.cached_timestamps = timestamps_per_game
        
        return self.cached_data[idx % self.cache_size], self.cached_targets[idx % self.cache_size], self.cached_timestamps[idx % self.cache_size]

def index_split(n_games):
    indices = np.arange(n_games)
    random.shuffle(indices)
    split_index = int(n_games // 1.1111111)
    return sorted(indices[:split_index]),sorted(indices[split_index:])

with open('data_stats.json', 'r') as file:
    data_stats = json.load(file)

dataset = LoLDatasetCache(data_stats['max_len'], data_stats['n_games'],80)
train_indices, test_indices = index_split(data_stats['n_games'])
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

k =0 
model = LinearRegression()
for i in range(ITERATIONS):
    it = 0
    for X, y,t in tqdm(train_loader):
        k+=1
        X = X.numpy()
        y = y.numpy()
        t = t.numpy()
        # flatten 0,1 dimensions
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        y = y.reshape(y.shape[0] * y.shape[1])
        t = t.reshape(t.shape[0] * t.shape[1])

        # find all rows in X where all values are 0
        mask = np.all(X == 0, axis=1)
        mask_y = (y == -1.0)
        mask = mask | mask_y
        X = X[~mask]
        y = y[~mask]
        t = t[~mask]
        
        
        
        
        print("SHAPES: ",X.shape, y.shape)
        model.fit(X, y)
        # 
        
        
        # prededict on train set
        print('Predicting on train set')
        y_pr = model.predict(X)
        accuracies_per_percent = np.zeros(101)
        sample_count = np.zeros(101)
        for l in range(len(y_pr)):
            accuracies_per_percent[t[l]] += (y_pr[l] > 0.5).astype(np.float32) == y[l]
            sample_count[t[l]] += 1
        sample_count[sample_count == 0] = 1
        accuracies_per_percent = accuracies_per_percent / sample_count
        #use plotly to plot the accuracies
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(101), y=accuracies_per_percent))
        fig.update_layout(title='Accuracy per time percentage TRAIN SET')
        #fig.write_image(os.path.join(GRAPHS_FOLDER, f'accuracy_{i}.png'))
        fig.show()   
            
        print('Train set MSE itr@{}: {}'.format(k, sklearn.metrics.mean_squared_error(y, y_pr)))
        print('Train set Accuracy at the end: {}'.format(((y_pr>0.5).astype(np.float32) ==  y).mean()))
        
        if k%10 == 1:
            y_te = []
            y_pr = []
            t_combined = []
            it2 = 0
            for X, y,t in test_loader:
                it2+=1
                if it2 > 1:
                    break
                X = X.numpy()
                y = y.numpy()
                t = t.numpy()
                
                X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
                y = y.reshape(y.shape[0] * y.shape[1])
                t = t.reshape(t.shape[0] * t.shape[1])
                
                mask = np.all(X == 0, axis=1)
                X = X[~mask]
                y = y[~mask]
                t = t[~mask]
                
                t_combined.append(t)
                y_te.append(y)
                y_pr.append(model.predict(X))
            
            t_combined = np.concatenate(t_combined)
            y_te = np.concatenate(y_te)
            y_pr = np.concatenate(y_pr)
            accuracies_per_percent = np.zeros(101) 
            sample_count = np.zeros(101)
            for l in range(len(y_pr)):
                accuracies_per_percent[t_combined[l]] += (y_pr[l] > 0.5).astype(np.float32) == y_te[l]
                sample_count[t_combined[l]] += 1
            sample_count[sample_count == 0] = 1
            accuracies_per_percent = accuracies_per_percent / sample_count
            #use plotly to plot the accuracies
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(101), y=accuracies_per_percent))
            fig.update_layout(title='VAL SET Accuracy per time percentage ')
            #fig.write_image(os.path.join(GRAPHS_FOLDER, f'accuracy_{i}.png'))
            fig.show()   
            print('VAL SET MSE itr@{}: {}'.format(k, sklearn.metrics.mean_squared_error(y_te, y_pr)))
            print('VAL SET Accuracy itr@{}: {}'.format(k, ((y_pr>0.5).astype(np.float32) ==  y_te).mean()))
            # save model
#%% 
y_te = []
y_pr = []
t_combined = []
for X, y,t in test_loader:
    X = X.numpy()
    y = y.numpy()
    t = t.numpy()
    
    X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    y = y.reshape(y.shape[0] * y.shape[1])
    t = t.reshape(t.shape[0] * t.shape[1])
    
    mask = np.all(X == 0, axis=1)
    X = X[~mask]
    y = y[~mask]
    t = t[~mask]
    
    t_combined.append(t)
    y_te.append(y)
    y_pr.append(model.predict(X))

t_combined = np.concatenate(t_combined)
y_te = np.concatenate(y_te)
y_pr = np.concatenate(y_pr)
accuracies_per_percent = np.zeros(101) 
sample_count = np.zeros(101)
for l in range(len(y_pr)):
    accuracies_per_percent[t_combined[l]] += (y_pr[l] > 0.5).astype(np.float32) == y_te[l]
    sample_count[t_combined[l]] += 1
sample_count[sample_count == 0] = 1
accuracies_per_percent = accuracies_per_percent / sample_count
#use plotly to plot the accuracies
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(101), y=accuracies_per_percent))
fig.update_layout(title='VAL SET Accuracy per time percentage ')
#fig.write_image(os.path.join(GRAPHS_FOLDER, f'accuracy_{i}.png'))
fig.show()   
print('VAL SET MSE itr@{}: {}'.format(k, sklearn.metrics.mean_squared_error(y_te, y_pr)))
print('VAL SET Accuracy itr@{}: {}'.format(k, ((y_pr>0.5).astype(np.float32) ==  y_te).mean()))
# save model
# save model

np.savetxt("accuracies_per_percent_linear.csv", accuracies_per_percent, delimiter=",", header="Accuracy", comments='')
#model.save_model(os.path.join(CHECKPOINTS_FOLDER, f'model_final.json'))

# y_pr = model.predict(xgb.DMatrix(x_te))
# print('MSE at the end: {}'.format(sklearn.metrics.mean_squared_error(y_te, y_pr)))
# print('Accuracy at the end: {}'.format(((y_pr>0.5).astype(np.float32) ==  y_te).mean()))

# show feature importance
xgb.plot_importance(model)

# print number of features
print('Number of features: {}'.format(len(model.get_score(importance_type='weight'))))
# %%
