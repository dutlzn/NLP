"""
数据集制作
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


"""
制作数据集步骤：
1、读取每个句子，然后分词，跳过停止词
2、根据先前得到的词典，分词转化成int值，不在词典中的就是unk，长度不够的就是pad
"""

class EmotionDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        return 0, 0

    def __len__(self):
        return 0