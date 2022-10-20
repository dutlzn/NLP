import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import jieba
from torchvision import transforms

def load_data_and_processing(data_path, stop_words_path, vic_dict_path, config):
    data_list = open(data_path, 'r', encoding='UTF-8').readlines()[1:]
    stop_words_list = open(stop_words_path, 'r', encoding='UTF-8').readlines()
    stop_words_list = [item.strip() for item in stop_words_list]
    stop_words_list.append('\t')
    stop_words_list.append(' ')

    vic_dict = open(vic_dict_path, 'r', encoding='UTF-8').readlines()
    vic_dict = {line.split(':')[0] : (int)(line.split(':')[1]) for line in vic_dict}
    datas = []
    labels = []

    max_seq_len =  0
    min_seq_len = 1000000

    for item in data_list:
        label = item[0]
        labels.append(label)
        content = item[2:].strip()
        seg_list = jieba.lcut(content)
        subset = []
        for seg_item in seg_list:
            if seg_item in stop_words_list:
                continue
            if len(subset) >= config.max_seq_len:
                break
            if seg_item in vic_dict.keys():
                subset.append(vic_dict[seg_item])
            else:
                subset.append(vic_dict["<UNK>"])
        if len(subset) < config.max_seq_len:
            subset += [vic_dict["<PAD>"] for i in range(config.max_seq_len - len(subset))]
        datas.append(subset)
    datas = np.array(datas).astype(np.long)
    labels = np.array(labels).astype(np.long)
    return datas, labels

class TextDatasets(Dataset):
    def __init__(self, data_path, stop_words_path, vic_dict_path, config):
        self.datas, self.labels = load_data_and_processing(
                data_path,
                stop_words_path,
                vic_dict_path,
                config)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]

    def __len__(self):
        assert len(self.datas) == len(self.labels)
        return len(self.datas)

if __name__ == '__main__':
    from configs import config

    data_path = './data/weibo_senti_100k.csv'
    stop_words_path = './data/hit_stopword'
    vic_dict_path = './data/dict'
    cfg = config()

    dataset = TextDatasets(data_path, stop_words_path, vic_dict_path, cfg)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data, label in train_loader:
        print(label)
        print(data)
        break
