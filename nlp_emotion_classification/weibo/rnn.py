import torch 
import torch.nn as nn 
import torch.nn.functional as F
from configs import config

class Rnn(nn.Module):
    def __init__(self, config):
        super(Rnn, self).__init__()
        self.embedder = nn.Embedding(config.n_vocab,
                                     config.embed_size)
        self.rnn = nn.LSTM(config.embed_size,
                           config.hidden_size,
                           config.layer_num,
                           bidirectional=True)
        self.max_pool = nn.MaxPool1d(config.max_seq_len)
        self.fc = nn.Linear(config.hidden_size*2+config.embed_size,
                            config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedder(x)
        out, _ = self.rnn(x)
        x = torch.cat([out, x], dim=2)
        x = x.permute(0, 2, 1)
        x = self.max_pool(x).reshape(x.size()[0], -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

