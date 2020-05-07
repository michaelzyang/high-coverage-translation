import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryModel(nn.Module):
    """ filtering model """
    def __init__(self, src_word_num, src_word_dim, tgt_word_num, tgt_word_dim, hidden_dim, dropout):
        super(BinaryModel, self).__init__()
        self.src_embed = nn.Embedding(src_word_num, src_word_dim, padding_idx=0)
        self.src_rnn = nn.LSTM(input_size=src_word_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.tgt_embed = nn.Embedding(tgt_word_num, tgt_word_dim, padding_idx=0)
        self.tgt_rnn = nn.LSTM(input_size=tgt_word_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.pos_embed = nn.Embedding(500, tgt_word_dim)
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(nn.Linear(5 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        batch_size = tgt.size(0)
        pos = torch.arange(batch_size, dtype=tgt.dtype, device=tgt.device)
        # encode source sentence
        src = self.src_embed(src)
        _, (src, _) = self.src_rnn(src) # (num_layers * num_directions, batch, hidden_size)
        src = src.transpose(0, 1).contiguous().view(1, -1) # (batch_size, hidden_size * 2)
        src = src.expand(batch_size, -1)
        # encode candidates
        tgt = self.tgt_embed(tgt)
        _, (tgt, _) = self.tgt_rnn(tgt) # (num_layers * num_directions, batch, hidden_size)
        tgt = tgt.transpose(0, 1).contiguous().view(batch_size, -1) # (batch_size, hidden_size * 2)
        # encoder candidate position
        pos = self.pos_embed(pos)
        # decode
        x = self.mlp(self.dropout(torch.cat((src, tgt, pos), dim=1)))
        return x.squeeze()