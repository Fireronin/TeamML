#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
data = np.load("data.npy", allow_pickle=True)
labels = np.load("labels.npy", allow_pickle=True)
# %%
# data.shape (100, 215, 1279)
# labels.shape (100, 215, 2)

#%%
# split into train and test using sklearn
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2)



# %%
# create a dataset class for transformers
# This data set will be made of out a list of sequences of variable length but they are padded to the max length

# max length of a sequence
max_len = train_x.shape[1]

#%%

class LolDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        # get the df at idx
        x = self.x[idx]
        y = self.y[idx]
        
        # convert to tensor
        x = torch.tensor(x)
        y = torch.tensor(y)
        
        
        # find a moment where y for both is 0 
        sum_y = y.sum()
        # 1 till from 0 to sum_y-1, rest is 0
        # mask = torch.zeros_like(y)
        # mask[0:sum_y] = 1        
        
        return x, y,sum_y
    
#%%

# create the dataset
train_dataset = LolDataset(train_x, train_y)
test_dataset = LolDataset(test_x, test_y)

#%%
batch_size = 10
# create a dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#%%
# create a transformer decoder only

class LolTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_size, nhead=1)
        
        self.transformer = nn.TransformerDecoder(self.decoder_layer, num_layers=4)
        self.memory = torch.zeros((batch_size, max_len, input_size)).to(device)
        self.linear = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x,valid_length):
        mask = torch.zeros((batch_size, max_len, max_len)).to(device)
        # use torch function to create mask
        mask = torch.nn.Transformer.generate_square_subsequent_mask(self, valid_length)
        x = self.transformer(x, self.memory,)
        x = self.linear(self.memory)
        x = self.softmax(x)
        return x
        
    
#%%
# create a model
model = LolTransformer(input_size=1279, output_size=2).to(device)

#%%
# create an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%%

# create a loss function
criterion = nn.CrossEntropyLoss()

#%%

# create a training loop
epochs = 10
for epoch in range(epochs):
    # set model to train mode
    model.train()
    
    # create a list to store the losses
    losses = []
    
    # loop over the data
    for x, y in train_loader:
        # send to device
        x = x.to(device)
        y = y.to(device)
        
        # zero the gradients
        optimizer.zero_grad()
        
        # forward pass
        y_pred = model(x)
        
        # compute the loss
        loss = criterion(y_pred, y)
        
        # append the loss to the list
        losses.append(loss.item())
        
        # backward pass
        loss.backward()
        
        # update the parameters
        optimizer.step()
        
    # print the loss
    print(f"Epoch: {epoch}, Loss: {np.mean(losses)}")
    
# %%
