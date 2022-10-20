import random

import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
from configs import config 

class Embedder(nn.Module):
    def __init__(self, config):
        super(Embedder, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab,
                                      config.embed_size,
                                      padding_idx=config.n_vocab - 1)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoder(nn.Module):
    def __init__(self, config):
        super(PositionalEncoder, self).__init__()
        self.d_model = config.embed_size
        pe = torch.zeros(config.max_seq_len, config.embed_size).to(config.devices)

        for pos in range(config.max_seq_len):
            for i in range(0, config.embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i+1)) / self.d_model)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False
        self.pe.to(config.devices)
    def forward(self, x):
        return math.sqrt(self.d_model) * x + self.pe


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.q_linear = nn.Linear(config.embed_size, config.embed_size)
        self.k_linear = nn.Linear(config.embed_size, config.embed_size)
        self.v_linear = nn.Linear(config.embed_size, config.embed_size)

        self.out = nn.Linear(config.embed_size, config.embed_size)
        self.d_model = config.embed_size

    def forward(self, q, k, v, mask):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model)

        # 为什么是1不是0
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)

        normlized_weights = F.softmax(weights, dim=-1)
        output = torch.matmul(normlized_weights, v)
        output = self.out(output)
        return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()

        self.l1 = nn.Linear(config.embed_size, config.d_ff)
        self.dropout = nn.Dropout(config.dropout)
        self.l2 = nn.Linear(config.d_ff, config.embed_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(F.relu(x))
        x = self.l2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()

        self.n1 = nn.LayerNorm(config.embed_size)
        self.n2 = nn.LayerNorm(config.embed_size)

        self.attn = Attention(config)

        self.ff = FeedForward(config)

        self.d1 = nn.Dropout(config.dropout)
        self.d2 = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        x_norm = self.n1(x)
        output, _ = self.attn(x_norm, x_norm, x_norm, mask)
        x2 = x + self.d1(output)
        x_norm2 = self.n2(x2)
        output = x2 + self.d2(self.ff(x_norm2))
        return output, _
    
class ClassificationHead(nn.Module):
    def __init__(self, config):
        super(ClassificationHead, self).__init__()

        self.linear = nn.Linear(config.embed_size, config.embed_size)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]
        out = self.linear(x0)
        return out


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.n1 = Embedder(config)
        self.n2 = PositionalEncoder(config)
        self.n3 = TransformerBlock(config)
        self.n4 = TransformerBlock(config)
        self.n5 = ClassificationHead(config)

    def forward(self, x, mask):
        x1 = self.n1(x)
        x2 = self.n2(x1)
        x3_1, norm_weights1 = self.n3(x2, mask)
        x3_2, norm_weights2 = self.n4(x3_1, mask)
        x4 = self.n5(x3_2)
        return x4, norm_weights1, norm_weights2

if __name__ == '__main__':
    # x = torch.tensor([random.randint(0, 100) for _ in range(5)])
    x = torch.tensor([10, 20, 50, 50, 1])
    print(x)
    x = (x != 50)
    print(x)
    weights = torch.tensor([1,2,3,4,5])
    weights = weights.masked_fill(x == 0, 1e8)
    print(weights)
