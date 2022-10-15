import pandas as pd
import numpy as np
import jieba
"""
制作词典步骤：
1、每个评论进行分词，过滤掉停止词
2、制作词频字典，统计分词之后每个词对应出现的次数
3、按照词频进行Top-K从大到小排序
4、对应词频折算成0~K索引，然后保存词典
"""

data_path = '../data/ChnSentiCorp_htl_all.csv'
stop_words_path = '../data/hit_stopword'
UNK = "<UNK>"
PAD = "<PAD>"

pd_all = pd.read_csv(data_path)
print('评论总数:{}'.format(pd_all.shape[0]))
print('正面评论总数:{}'.format(pd_all[pd_all.label == 1].shape[0]))
print('负面评论总数:{}'.format(pd_all[pd_all.label == 0].shape[0]))

data_list = open(data_path, encoding='UTF-8').readlines()[1:]
stop_words_list = open(stop_words_path, encoding='UTF-8').readlines()
stop_words_list = [word.strip() for word in stop_words_list]

vic_dict = {}

for item in data_list:
    label = item[0]
    context = item[2:].strip()
    seg_list = jieba.cut(context, cut_all=False)
    for seg_item in seg_list:
        if seg_item in stop_words_list:
            continue
        if seg_item in vic_dict.keys():
            vic_dict[seg_item] += 1
        else:
            vic_dict[seg_item] = 1

top_k = 1000
vic_dict = sorted([item for item in vic_dict.items()], key=lambda x:x[1], reverse=True)[:top_k]
vic_dict = [[item[0], idx] for idx, item in enumerate(vic_dict)]

vic_dict.append([UNK, top_k])
vic_dict.append([PAD, top_k + 1])

ff = open("../data/dict", 'w', encoding='UTF-8')
for item in vic_dict:
    ff.writelines("{}:{}\n".format(item[0], item[1]))
ff.close()

