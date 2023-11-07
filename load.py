#%%
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
#%%

folder_name = "new2"



# read
files_list = os.listdir(folder_name)
files_list.sort()

number_to_read = os.listdir(folder_name).__len__()

# read first n files
files_list = files_list[:number_to_read]

df = pd.DataFrame()


# combine all files in the list
for file in files_list:
    df = pd.concat([df, pd.read_csv(os.path.join(folder_name, file))])

df.head()

# read 

#%%
df.describe()

# using variable counts divide columns into categorical and numerical
#%%

# get column names
column_names = df.columns

# get column counts but only unique values
column_counts = df.nunique()

# sort column names by counts
column_counts.sort_values(inplace=True)

# for each column create a map from value to index that will be used to convert to one hot encoding
maps = {}
for column in column_names:
    # get the unique values
    unique_values = df[column].unique()
    # create a map from value to index
    maps[column] = {value: index for index, value in enumerate(unique_values)}


# 
import matplotlib.pyplot as plt

# show histogram of counts
plt.hist(column_counts, bins=100)
plt.show()


# %%

cut_off = 1000

# the plan is to later use this into a transformer that will 

df_list = []
winning_team = {100: 0, 200: 0}
# get the winning team
    
    
for file in tqdm(files_list):
    df_list.append(pd.read_csv(os.path.join(folder_name, file)))
    # count the number of times each team won
    winning_team[df_list[-1]['winning_team'].iloc[0]] += 1

print(winning_team)
    
for df in df_list:
    df.drop(columns=['matchId'], inplace=True)


# %%
# drop match id column


# find columns with less than cut_off unique values
columns_to_encode = column_counts[column_counts < cut_off].index

# find columns with more than cut_off unique values
columns_to_scale = column_counts[column_counts >= cut_off].index

# remove match id from columns to scale
columns_to_scale = columns_to_scale.drop('matchId')

#%% 
# compute mean and std for columns_to_scale
mean = df[columns_to_scale].mean()
std = df[columns_to_scale].std()


for df in df_list:
    df[columns_to_scale] = (df[columns_to_scale] - mean) / std

# %%
# convert columns_to_encode to categorical
for df in tqdm(df_list):
    for column in columns_to_encode:
        # maps[column] is a map from value to index
        # convert to categorical
        df[column] = df[column].map(maps[column])
# convert categorical to one hot encoding
#df = pd.get_dummies(df, columns=columns_to_encode)
#%%
# print types
# force to print all and not truncate
pd.set_option('display.max_columns', None)
print(list(df_list[0].dtypes))
#%%
#df.fillna(-10, inplace=True)
for df in tqdm(df_list):
    df.fillna(-10, inplace=True)

#%% 
# split into train and test manually (no sklearn)

examples = len(df_list)

split = 0.8

train = df.iloc[:int(examples * split)]
test = df.iloc[int(examples * split):]
#%%

train_y = []
train_x = []
for df in tqdm(df_list):
    train_y.append(df['winning_team'])
    train_x.append(df.drop(columns=['winning_team']))
    
test_y =[]
test_x = []

for df in tqdm(df_list):
    test_y.append(df['winning_team'])
    test_x.append(df.drop(columns=['winning_team']))

#%%
sum_w = 0
for train in test_y:
    sum_w += np.array(train).sum()
print(sum_w)

#%% 
# for all x columns use column_counts to convert to one hot encoding
# for all y columns use column_counts to convert to one hot encoding
# pd.get_dummies won't work because it will create different columns for different dataframes


def df_to_one_hot(df, column_counts,columns_to_encode):
    # create a new dataframe
    new_df = pd.DataFrame()
    # for each column in df
    # if column is in columns_to_encode
    # convert to categorical and then to one hot encoding use column_counts to get the number of categories
    # else
    # just copy the column
    for column in df.columns:
        if column in columns_to_encode:
            new_df = pd.concat([new_df, pd.get_dummies(df[column], columns=column_counts[column])], axis=1)
        else:
            new_df = pd.concat([new_df, df[column]], axis=1)
    

    return new_df

for i in tqdm(range(len(train_x))):
    train_x[i] = df_to_one_hot(train_x[i], column_counts, columns_to_encode)
    test_x[i] = df_to_one_hot(test_x[i], column_counts, columns_to_encode)
    
#%%
# convert boolean to int true = 1 false = 0
for i in range(len(train_y)):
    train_y[i] = train_y[i].astype(int)
    test_y[i] = test_y[i].astype(int)


# print first value of train_y for all dataframes
for i in range(len(train_y)):
    print(train_y[i].iloc[0])

# %% import all torch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

#%% create a dataset class for transformers
# This data set will be made of out a list of sequences of variable length
# each sequence will be padded to the maximum length of the sequence

# max length of a sequence
max_len = -1
for df in df_list:
    max_len = max(max_len, df.shape[0])
    
#%%

class LolDataset(torch.utils.data.Dataset):
    def __init__(self, df_list, max_len):
        self.df_list = df_list
        self.max_len = max_len
        
    def __len__(self):
        return len(self.df_list)
    
    def __getitem__(self, idx):
        # get the df at idx
        df = self.df_list[idx]
        
        # get the length of the df
        length = df.shape[0]
        
        # convert to tensor
        df = torch.tensor(df.values)
        
        # pad the sequence
        df = F.pad(df, (0, 0, 0, self.max_len - length))
        
        return df, length
    
#%%
# create the dataset
lol_dataset = LolDataset(df_list, max_len)

#%%
# create a dataloader
lol_loader = torch.utils.data.DataLoader(lol_dataset, batch_size=1, shuffle=True)

#%%
# create a transformer
class LolTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, output_size, dropout):
        super(LolTransformer, self).__init__()
        
        # create embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # create positional encoding
        self.positional_encoding = nn.Embedding(max_len, hidden_size)
        
        # create transformer
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dim_feedforward=hidden_size, dropout=dropout)
        
        # create linear layer
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, length):
        # get the batch size
        batch_size = x.shape[1]
        
        # create the positional encoding
        positional_encoding = self.positional_encoding(torch.arange(length).repeat(batch_size, 1).to(device))
        
        # add positional encoding to x
        x = x + positional_encoding
        
        # create mask
        mask = self.transformer.generate_square_subsequent_mask(length).to(device)
        
        # create embedding
        x = self.embedding(x)
        
        # transpose x
        x = x.permute(1, 0, 2)
        
        # transformer
        x = self.transformer(x, x, tgt_mask=mask)
        
        # transpose x
        x = x.permute(1, 0, 2)
        
        # linear
        x = self.linear(x)
        
        return x
    
    
#%%
# create the transformer
input_size = df_list[0].shape[1]
hidden_size = 512
num_layers = 3
num_heads = 8
output_size = 1
dropout = 0.1

lol_transformer = LolTransformer(input_size, hidden_size, num_layers, num_heads, output_size, dropout)

#%%
# create the optimizer
optimizer = optim.Adam(lol_transformer.parameters(), lr=0.001)

#%%
# create the loss function
loss_function = nn.MSELoss()

#

# %%

# create the training loop
epochs = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lol_transformer.to(device)

for epoch in tqdm(range(epochs)):
    for x, length in lol_loader:
        # send to device
        x = x.to(device)
        
        # get the output
        output = lol_transformer(x, length)
        
        # get the target
        target = x[:, :, 0].unsqueeze(2)
        
        # compute the loss
        loss = loss_function(output, target)
        
        # zero the gradients
        optimizer.zero_grad()
        
        # backpropagate
        loss.backward()
        
        # update the weights
        optimizer.step()
        
    print(f'Epoch: {epoch} Loss: {loss.item()}')
# %%
