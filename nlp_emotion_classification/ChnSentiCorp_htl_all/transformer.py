"""
Transformer相关实现
"""
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from configs import config

class Embedder(nn.Module):
    def __init__(self, config):
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(config.n_vocab,
                                  config.embed_size,
                                  padding_idx=config.n_vocab - 1)

    def forward(self, x):
        x = x.to(torch.int64)
        return self.embed(x)

"""
位置编码
"""
class PositionalEncoder(nn.Module):
    def __init__(self, config):
        super(PositionalEncoder, self).__init__()
        self.d_model = config.embed_size # 单词向量的维度
        # 创建根据单词的顺序(pos)和填入向量的维度的位置(i)确定的值的表pe
        # pe = torch.zeros(config.embed_size, config.max_seq_len)
        pe = torch.zeros(config.max_seq_len, config.embed_size)
        for pos in range(config.max_seq_len):
            for i in range(0, config.embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i) / config.embed_size)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2*i) / config.embed_size)))

        # 将作为小批量维度的维度添加到pe的开头
        self.pe = pe.unsqueeze(0)

        # 关闭梯度计算
        self.pe.requires_grad = False

    def forward(self, x):
        # x比pe要小，因此要将其放大
        res = math.sqrt(self.d_model) * x + self.pe
        return res

"""
单头注意力
"""
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.q_linear = nn.Linear(config.embed_size, config.embed_size)
        self.k_linear = nn.Linear(config.embed_size, config.embed_size)
        self.v_linear = nn.Linear(config.embed_size, config.embed_size)

        # 输出时使用的全连接层
        self.out = nn.Linear(config.embed_size, config.embed_size)

        # 用于调整attention大小的变量
        self.d_k = config.embed_size

    def forward(self, q, k, v, mask):
        k = self.k_linear(k)
        q = self.q_linear(q)
        v = self.v_linear(v)

        weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)
        normlized_weights = F.softmax(weights, dim=1)
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
        self.norm_1 = nn.LayerNorm(config.embed_size)
        self.norm_2 = nn.LayerNorm(config.embed_size)

        self.attn = Attention(config)
        self.ff = FeedForward(config)

        self.dropout_1 = nn.Dropout(config.dropout)
        self.dropout_2 = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        x_normlized = self.norm_1(x)
        output, normlized_weights = self.attn(x_normlized, x_normlized, x_normlized, mask)
        x2 = x + self.dropout_1(output)

        x_normlized2 = self.norm_2(x2)
        output = x2 + self.dropout_2(self.ff(x_normlized2))
        return output


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(config.embed_size * config.max_seq_len, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.reshape(x.size()[0], -1)
        x = self.linear(x)
        x = F.softmax(x)
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.net1 = Embedder(config)
        self.net2 = PositionalEncoder(config)
        self.net3_1 = TransformerBlock(config)
        self.net3_2 = TransformerBlock(config)
        self.net4 = ClassificationHead(config)

    def forward(self, x, mask):
        x1 = self.net1(x)
        x2 = self.net2(x1)
        x3_1 = self.net3_1(x2, mask)
        x3_2 = self.net3_2(x3_1, mask)
        x4 = self.net4(x3_2)
        return x4

if __name__ == '__main__':
    cfg = config()

    # input_pad = 1
    # x = torch.tensor([random.randint(0, 256) for _ in range(256)])
    # input_mask = (x != input_pad)
    # net = Transformer(config=cfg)
    # y = net(x, input_mask)
    # print(y.size())