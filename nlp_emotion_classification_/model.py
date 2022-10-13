import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


"""
模型类定义
1 100
1 100 * 4 词嵌入
"""
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


if __name__ == '__main__':
    from configs import config
    cfg = config()
    cfg.pad_size = 640
    model = Model(config=cfg)
    input_tensor = torch.tensor([i for i in range(640)]).reshape([1, 640])
    output_tensor = model(input_tensor)
    print(output_tensor)