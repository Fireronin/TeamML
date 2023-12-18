import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class TransformerModel(nn.Module):
    def __init__(self, output_dim, nhead, nhid, nlayers, ngame_cont, nteam_cont, nplayer_cont, nitems, nchampions, nrunes, game_dim, team_dim, player_dim, item_dim, champion_dim, runes_dim, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None

        input_dim = game_dim + item_dim + 2 * team_dim + 10 * (player_dim + champion_dim + runes_dim)

        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = TransformerEncoderLayer(input_dim, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.game_linear = nn.Linear(ngame_cont, game_dim)
        self.team_linear = nn.Linear(nteam_cont, team_dim)  
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

        self.decoder = nn.Linear(input_dim, output_dim) 

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src_game = src[:, :, :self.ngame_cont]
        src_item = src[:, :, self.ngame_cont:(self.ngame_cont + 1)]
        src_teams = [src[:, :, (self.ngame_cont + 1 + i * self.nteam_cont):(self.ngame_cont + 1 + (i + 1) * self.nteam_cont)] for i in range(2)]
        src_player_linears = [src[:, :, (self.ngame_cont + 1 + 2 * self.nteam_cont + i * (self.nplayer_cont + 10)):(self.ngame_cont + 1 + 2 * self.nteam_cont) + (i + 1) * (self.nplayer_cont + 10) - 10] for i in range(10)]
        src_player_champions = [src[:, :, (self.ngame_cont + 1 + 2 * self.nteam_cont + (i + 1) * (self.nplayer_cont + 10) - 10):(self.ngame_cont + 1 + 2 * self.nteam_cont) + (i + 1) * (self.nplayer_cont + 10) - 9] for i in range(10)]
        src_player_runes = [src[:, :, (self.ngame_cont + 1 + 2 * self.nteam_cont + (i + 1) * (self.nplayer_cont + 10) - 9):(self.ngame_cont + 1 + 2 * self.nteam_cont) + (i + 1) * (self.nplayer_cont + 10)] for i in range(10)]

        src_game = self.game_linear(src_game)
        src_item = self.item_embedding(src_item) * math.sqrt(self.item_dim)
        src_teams = [self.team_linear(team) for team in src_teams]
        src_player_linears = [self.player_linear(player) for player in src_player_linears]
        src_player_champions = [self.champion_embedding(player) * math.sqrt(self.champion_dim) for player in src_player_champions]
        src_player_runes = [self.runes_embedding(rune) * math.sqrt(self.runes_dim) for runes in src_player_runes for rune in runes]

        src = [src_game, src_item, src_teams[0], src_teams[1]]
        src.extend([item for sublist in zip(src_player_linears, src_player_champions, src_player_runes) for item in sublist])

        src = torch.cat(src, dim=-1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output[-1])
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
