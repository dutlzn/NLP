import numpy as np
import pandas as pd
import jieba

data_path = './data/weibo_senti_100k.csv'
stop_words_path = './data/hit_stopword'

data_list = open(data_path, 'r', encoding='UTF-8').readlines()[1:]
stop_words_list = open(stop_words_path, 'r', encoding='UTF-8').readlines()
stop_words_list = [item.strip() for item in stop_words_list]
stop_words_list.append("\t")
stop_words_list.append(" ")

vic_dict = {}

for item in data_list:
    label = item[0]
    content = item[2:].strip()
    seg_list = jieba.lcut(content, cut_all=False)
    for seg_item in seg_list:
        if seg_item in stop_words_list:
            continue

        if seg_item in vic_dict.keys():
            vic_dict[seg_item] += 1
        else:
            vic_dict[seg_item] = 0


top_K = 1000
vic_dict = sorted([item for item in vic_dict.items()],
                  key=lambda x:x[1],
                  reverse=True)[:top_K]
vic_dict = {item[0]: idx for idx, item in enumerate(vic_dict)}
# 未出现在词典中的单词
UNK = "<UNK>"
# 当句子分词长度不够时候的填充
PAD = "<PAD>"

vic_dict.update({UNK: len(vic_dict),
                 PAD: len(vic_dict)+1})


ff = open('./data/dict', 'w', encoding='UTF-8')

for item in vic_dict.items():
    ff.writelines("{}:{}\n".format(item[0], item[1]))

ff.close()