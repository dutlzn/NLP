import torch
import torch.nn as nn
import torch.nn.functional as f
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab,
                                      config.embed_size,
                                      padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed_size,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)

        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size*2 + config.embed_size,
                            config.num_classes)

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.to(torch.int64)
        embed = self.embedding(x) # batch_size * seq_len * embed_size
        out, _ = self.lstm(embed)
        out = torch.cat([out, embed], 2)
        out = f.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).reshape(out.size()[0], -1)
        out = self.fc(out)
        out = self.softmax(out)
        return out