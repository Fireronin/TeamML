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

df = pl.read_parquet('transformed_data/timeline_0.parquet')

# for each column compute the number of unique values, mean, std and max
n_unique = []
mean = []
std = []
max_ = []
for col in df.columns:
    if col == 'matchId':
        continue
    n_unique.append(df[col].n_unique())
    mean.append(df[col].mean())
    std.append(df[col].std())
    max_.append(df[col].max())
    

# print the results
print(f'Number of unique values: {n_unique}')
print(f'Mean: {mean}')
print(f'Std: {std}')
print(f'Max: {max_}')



# if std is 0, replace it with 1
std = [s if s != 0 else 1 for s in std]

print(f'Device: {DEVICE}')
games = []
grouped = df.group_by(['matchId'])
# print number of games
print(f'Number of games: {grouped.count().shape[0]}')
# take only first 1000 games


for name, group in grouped:
    group = group.drop('matchId')
    games.append(torch.from_numpy(group.to_numpy()))
    
print(games[0].shape)
# %%

#Split the data into train and test

TRAIN_SIZE = 700

games_train = games[:TRAIN_SIZE]
games_test = games[TRAIN_SIZE:]

# [game,sequence,features]
# -> [game*sequence,features]

def prepare_data(games):
    flatten_games = torch.cat(games, dim=0)

    X = flatten_games[:, :-1]
    y = flatten_games[:, -1]

    y = (y/100)-1

    x_numpy = X.numpy()
    y_numpy = y.numpy().astype(int)
    # convert y to one hot
    y_numpy = np.eye(2)[y_numpy]
    return x_numpy, y_numpy

x_numpy, y_numpy = prepare_data(games_train)
x_numpy_test, y_numpy_test = prepare_data(games_test)

# Import XGBoost and train tree model
#%%
import xgboost as xgb

# Create the DMatrix
dtrain = xgb.DMatrix(x_numpy, label=y_numpy)

# Create the parameter dictionary 2 classes, 701 features

param = {'objective': 'binary:logistic',  # Objective is binary classification
        'booster':'dart',
        'max_depth': 15,  # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
        'learning_rate': 0.1,  # Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features.
        'subsample': 0.2,  # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting.
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'n_estimators': 100,  # Number of trees to fit.
        'reg_alpha': 0.01,  # L1 regularization term on weights. Increasing this value will make model more conservative.
        'reg_lambda': 1,  # L2 regularization term on weights. Increasing this value will make model more conservative.
        'scale_pos_weight': 1,  # Control the balance of positive and negative weights, useful for unbalanced classes.
        'random_state': 42,  # Random number seed. It can be used for producing reproducible results and also for parameter tuning.
        'eval_metric': 'logloss',  # Evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking)
        "device": 'cpu'}

# Train the model
bst = xgb.train(param, dtrain)


# Create the DMatrix: dtest
dtest = xgb.DMatrix(x_numpy_test)

# Predict the labels of the test set: preds
preds = bst.predict(dtest)

# Compute the accuracy: accuracy
# preds shape (n_samples, 2)
# y_numpy_test shape (n_samples, 2)

accuracy = (preds.argmax(axis=1) == y_numpy_test.argmax(axis=1)).mean()


print(f'accuracy: {accuracy}')

# mse
mse = ((preds - y_numpy_test) ** 2).mean()
print(f'mse: {mse}')




# %%

# for each game in games_test
# flatten the game, predict the outcome at every time step
# then average accuracy of predictions across all games
# using plotly, plot the accuracy of the predictions at each time step

max_game_len = max([game.shape[0] for game in games_test])

accuracy_per_timestep = np.zeros(max_game_len)
added_preds = np.zeros(max_game_len)

for game in games_test:
    x,y = prepare_data([game])
    dtest = xgb.DMatrix(x)
    preds = bst.predict(dtest)
    accuracy = (preds.argmax(axis=1) == y.argmax(axis=1))
    accuracy_per_timestep[:game.shape[0]] += accuracy
    added_preds[:game.shape[0]] += 1
    
# 
accuracy_per_timestep = accuracy_per_timestep / added_preds

trace0 = go.Scatter(
    y=accuracy_per_timestep,
    mode='lines',
    name='accuracy'
)

added_preds_normalized = added_preds / added_preds.max()

trace1 = go.Scatter(
    y=added_preds_normalized,
    mode='lines',
    name='added_preds'
)

fig = go.Figure(data=[trace0,trace1])
fig.show()
# %%
