"""
数据集制作
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from configs import config

"""
制作数据集步骤：
1、读取每个句子，然后分词，跳过停止词
2、根据先前得到的词典，分词转化成int值，不在词典中的就是unk，长度不够的就是pad
"""
"""
create datasets dataloader
"""
import jieba
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# data_path = '../data/weibo_senti_100k.csv'
# stop_words_path = '../data/hit_stopword'


def read_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path, encoding='utf-8').readlines()
    for item in dict_list:
        item = item.split(':')
        try:
            voc_dict[item[0]] = int(item[1].strip())
        except:
            print(item)
    return voc_dict


def load_data(data_path, data_stop_path):
    data_list = open(data_path, encoding='utf-8').readlines()[1:]
    stop_words = open(data_stop_path, encoding='utf-8').readlines()
    stop_words = [line.strip() for line in stop_words]
    stop_words.append(" ")
    stop_words.append("\n")
    voc_dict = {}
    data = []
    max_len_seq = 0
    # 随机打乱评论数据
    np.random.shuffle(data_list)
    for item in data_list[:1000]:
        label = item[0]
        content = item[2:].strip()
        seg_list = jieba.cut(content, cut_all=False)
        seg_res = []
        for seg_item in seg_list:
            if seg_item in stop_words:
                continue
            seg_res.append(seg_item)
            if seg_item in voc_dict.keys():
                voc_dict[seg_item] += 1
            else:
                voc_dict[seg_item] = 1

        if len(seg_res) > max_len_seq:
            max_len_seq = len(seg_res)
        data.append([label, seg_res])
        # print('max_len_seq:', max_len_seq)
    return data


class text_CLS(Dataset):
    """
    input:
        voc_dict_path: 生成词典路径
        data_path: 原始数据路径
        data_stop_path: 停止词路径
    """

    def __init__(self, config, voc_dict_path, data_path, data_stop_path):
        self.data_path = data_path
        self.voc_dict = read_dict(voc_dict_path)
        self.data_stop_path = data_stop_path
        self.data = load_data(self.data_path, self.data_stop_path)
        self.max_seq_len = config.max_seq_len
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = int(data[0])
        word_list = data[1]
        input_idx = []
        for word in word_list:
            if word in self.voc_dict.keys():
                # 输入训练数据为单词在词典中的索引
                input_idx.append(self.voc_dict[word])
            else:
                # 如果单词不在词典中，就直接"<UNK>"
                input_idx.append(self.voc_dict["<UNK>"])
        # 如果序列长度不够就补齐

        if len(input_idx) < self.max_seq_len:
            input_idx += [self.voc_dict["<PAD>"] for _ in range(self.max_seq_len - len(input_idx))]
        elif len(input_idx) > self.max_seq_len:
            input_idx = input_idx[:self.max_seq_len]

        input_idx = np.array(input_idx).astype(np.long)
        return label, input_idx

"""
自定义数据加载器
"""
def data_loader(dataset, config):
    # dataset = text_CLS(dict_path, data_path, data_stop_path)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle)


if __name__ == '__main__':
    from configs import config
    cfg = config()
    data_path = '../data/weibo_senti_100k.csv'
    stop_words_path = '../data/hit_stopword'
    vic_path = '../data/dict'
    dataset = text_CLS(cfg, vic_path, data_path, stop_words_path)
    print(dataset.max_seq_len)
    cfg = config()
    loader = data_loader(dataset, cfg)
    print(loader)





