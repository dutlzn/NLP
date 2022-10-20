import numpy as np
import jieba

data_path = '../data/weibo_senti_100k.csv'
stop_path = '../data/hit_stopword'
data_list = open(data_path, 'r', encoding='UTF-8').readlines()[1:]
stop_list = open(stop_path, 'r', encoding='UTF-8').readlines()
stop_list = [item.strip() for item in stop_list]
stop_list.append(' ')
stop_list.append('\t')

vic_dict = {}
for item in data_list:
    label = item[0]
    content = item[2:].strip()
    seg_list = jieba.lcut(content)
    for seg_item in seg_list:
        if seg_item in stop_list:
            continue

        if seg_item in vic_dict.keys():
            vic_dict[seg_item] += 1
        else:
            vic_dict[seg_item] = 1

top_K = 1000
vic_dict = sorted([item for item in vic_dict],
                  lambda x:x[1],
                  reverse=True)[:top_K]
vic_dict = {item[0]: idx for idx, item in enumerate(vic_dict)}


