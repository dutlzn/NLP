"""
data processing:
    create dict include emotion label and review
"""
import pandas as pd
import jieba

data_path = r'./data/weibo_senti_100k.csv'
stop_path = r'./data/hit_stopword'

voc_dict = {}
min_seq = 1
top_k = 1000

pd_all = pd.read_csv(data_path)
stop_words = open(stop_path).readlines()
stop_words = [line.strip() for line in stop_words]
stop_words.append(" ")
stop_words.append("\n")


# data visual
print(pd_all.head(3))
print('评论数目: {}'.format(pd_all.shape[0]))
print('评论正面数目: {}'.format(pd_all[pd_all.label == 0].shape[0]))
print('评论负面数目: {}'.format(pd_all[pd_all.label == 1].shape[0]))
print('-------------------------------')

# get dict by jieba & hit_stopwords
data_list = open(data_path).readlines()[1:]
for item in data_list:
    label = item[0]
    content = item[2:].strip()
    seg_list = jieba.cut(content, cut_all=False)
    for seg_item in seg_list:
        if seg_item in stop_words:
            continue
        if seg_item in voc_dict.keys():
            voc_dict[seg_item] += 1
        else:
            voc_dict[seg_item] = 1

# sort
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                  key=lambda x : x[1],
                  reverse=True)[:top_k]
print(voc_list[:20])


UNK = '<UNK>'
PAD = '<PAD>'

voc_dict = {data[0]: idx for idx, data in enumerate(voc_list)}
voc_dict.update({UNK: len(voc_dict), PAD: len(voc_dict) + 1})

ff = open('./data/dict', 'w')

for item in voc_dict.items():
    ff.writelines("{},{}\n".format(item[0], item[1]))
