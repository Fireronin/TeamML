import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class TransformerModel(nn.Module):
    def __init__(self, output_dim, nhead, nlayers, ngame_cont, nteam_cont, nplayer_cont, nitems, nchampions, nrunes, game_dim, team_dim, player_dim, item_dim, champion_dim, runes_dim, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None

        # input_dim: scalar value representing the total dimensionality of the input
        input_dim = game_dim + item_dim + 2 * team_dim + 10 * (player_dim + champion_dim + 9 * runes_dim)

        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # Linear layers and embeddings for different parts of the input
        self.game_linear = nn.Linear(ngame_cont, game_dim)
        # self.team_linear = nn.Linear(nteam_cont, team_dim)  
        self.player_linear = nn.Linear(nplayer_cont, player_dim)  
        self.item_embedding = nn.Embedding(nitems, item_dim)
        self.champion_embedding = nn.Embedding(nchampions, champion_dim) 
        self.runes_embedding = nn.Embedding(nrunes, runes_dim) 

        self.ngame_cont = ngame_cont
        self.nteam_cont = nteam_cont
        self.nplayer_cont = nplayer_cont

        self.item_dim = item_dim
        self.champion_dim = champion_dim
        self.runes_dim = runes_dim

        
        self.input_dim = input_dim
       
        # 2 learnable parameters for batch norm
        # self.gamma = nn.Parameter(torch.ones(input_dim))
        # self.beta = nn.Parameter(torch.zeros(input_dim))
        
        

        self.batch_norm = nn.BatchNorm1d(input_dim)

        self.decoder = nn.Linear(input_dim, output_dim) 

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        device = src.device
        # src: input tensor of shape (batch_size, game_len, n_features)
        if self.src_mask is None or self.src_mask.size(0) != src.shape[1]:   
            mask = self._generate_square_subsequent_mask(src.shape[1]).to(device)
            self.src_mask = mask

        # Splitting the input tensor into different parts
        src_game = src[:, :, :self.ngame_cont]  # shape: (batch_size, game_len, ngame_cont)
        src_item = src[:, :, self.ngame_cont:(self.ngame_cont + 1)].squeeze(dim=-1)  # shape: (batch_size, game_len)
        # src_teams and src_player_* are lists of tensors
        # src_teams = [src[:, :, (self.ngame_cont + 1 + i * self.nteam_cont):(self.ngame_cont + 1 + (i + 1) * self.nteam_cont)] for i in range(2)]  # each tensor in the list has shape: (batch_size, game_len, nteam_cont)

        src_player_linears = [src[:, :, (self.ngame_cont + 1 + 2 * self.nteam_cont + i * (self.nplayer_cont + 10)):(self.ngame_cont + 1 + 2 * self.nteam_cont) + (i + 1) * (self.nplayer_cont + 10) - 10] for i in range(10)]  # each tensor in the list has shape: (batch_size, game_len, nplayer_cont)
        src_player_champions = [src[:, :, (self.ngame_cont + 1 + 2 * self.nteam_cont + (i + 1) * (self.nplayer_cont + 10) - 10):(self.ngame_cont + 1 + 2 * self.nteam_cont) + (i + 1) * (self.nplayer_cont + 10) - 9].squeeze(dim=-1) for i in range(10)]  # each tensor in the list has shape: (batch_size, game_len)
        src_player_runes = [src[:, :, (self.ngame_cont + 1 + 2 * self.nteam_cont + (i + 1) * (self.nplayer_cont + 10) - 9):(self.ngame_cont + 1 + 2 * self.nteam_cont) + (i + 1) * (self.nplayer_cont + 10)] for i in range(10)] # each tensor in the list has shape: (batch_size, game_len, 9)

        # print(f'src_game shape before embed: {src_game.shape}')
        # print(f'src_item shape before embed: {src_item.shape}')
        # print(f'src_player_linears len before embed: {len(src_player_linears)}')
        # for i in range(10):
        #     print(f'src_player_linears[{i}] shape before embed: {src_player_linears[i].shape}')
        # print(f'src_player_champions len before embed: {len(src_player_champions)}')
        # for i in range(10):
        #     print(f'src_player_champions[{i}] shape before embed: {src_player_champions[i].shape}')
        # print(f'src_player_runes len before embed: {len(src_player_runes)}')
        # for i in range(10):
        #     print(f'src_player_runes[{i}] shape before embed: {src_player_runes[i].shape}')

        # Applying linear layers and embeddings
        src_game = self.game_linear(src_game)  # shape: (batch_size, game_len, game_dim)
        src_item = self.item_embedding.cpu()(src_item.cpu().to(torch.int)).to(device) * math.sqrt(self.item_dim) # shape: (batch_size, game_len, item_dim)

        # src_teams = [self.team_linear(team) for team in src_teams]  # each tensor in the list has shape: (batch_size, game_len, team_dim)
        src_player_linears = [self.player_linear(player) for player in src_player_linears]  # each tensor in the list has shape: (batch_size, game_len, player_dim)
        src_player_champions = [self.champion_embedding.cpu()(player.cpu().to(torch.int)).to(device) * math.sqrt(self.champion_dim) for player in src_player_champions]  # each tensor in the list has shape: (batch_size, game_len, champion_dim)
        src_player_runes = [(self.runes_embedding.cpu()(runes.cpu().to(torch.int)).to(device) * math.sqrt(self.runes_dim)).reshape(runes.shape[:-1] + (-1,)) for runes in src_player_runes]  # each tensor in the list has shape: (batch_size, game_len, runes_dim)

        # print(f'src_game shape after embed: {src_game.shape}')
        # print(f'src_item shape after embed: {src_item.shape}')
        # print(f'src_player_linears len after embed: {len(src_player_linears)}')
        # for i in range(10):
        #     print(f'src_player_linears[{i}] shape after embed: {src_player_linears[i].shape}')
        # print(f'src_player_champions len after embed: {len(src_player_champions)}')
        # for i in range(10):
        #     print(f'src_player_champions[{i}] shape after embed: {src_player_champions[i].shape}')
        # print(f'src_player_runes len after embed: {len(src_player_runes)}')
        # for i in range(10):
        #     print(f'src_player_runes[{i}] shape after embed: {src_player_runes[i].shape}')

        # Concatenating all the parts
        src = [src_game, src_item]
        src.extend([item for sublist in zip(src_player_linears, src_player_champions, src_player_runes) for item in sublist])  # list of tensors

        src = torch.cat(src, dim=-1)  # shape: (batch_size, game_len, input_dim)

        # batch norm
        #print(f'src shape before batch norm: {src.shape}')
        
        #src = self.batch_norm(src)
        
        # functional batch norm
        # src = nn.functional.batch_norm(src, self.gamma, self.beta, training=True)
        # apply this to each batch
        # for i in range(src.shape[0]):
        #     src[i] = self.batch_norm(src[i])

        src_reshaped = src.view(-1, src.shape[2])

        # Apply batch normalization
        src_normalized = self.batch_norm(src_reshaped)

        # Reshape src back to its original shape
        src = src_normalized.view(src.shape)

        # print(f'src shape before positional encoding: {src.shape}')

        src = self.pos_encoder(src)  # shape: (batch_size, game_len, input_dim)

        # print(f'src shape after positional encoding: {src.shape}')
        # print(f'mask shape: {self.src_mask.shape}')

        output = self.transformer_encoder(src, self.src_mask)  # shape: (batch_size, game_len, input_dim)

        # print(f'output shape after encoding: {output.shape}')

        output = self.decoder(output)  # shape: (batch_size, game_len, output_dim)
        output = nn.functional.softmax(output, dim=-1)

        # print(f'output shape after decoding: {output.shape}')

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe: tensor of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # position: tensor of shape (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term: tensor of shape (d_model // 2,)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2]: tensor of shape (max_len, d_model // 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2]: tensor of shape (max_len, d_model // 2)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe: tensor of shape (1, max_len, d_model) after unsqueeze
        pe = pe.unsqueeze(0)

        # print(f'pe shape: {pe.shape}')

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: input tensor of shape (batch_size, game_len, d_model)
        # self.pe[:x.size(0), :]: tensor of shape (1, game_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        # output tensor of shape (batch_size, game_len, d_model) after dropout
        return self.dropout(x)
