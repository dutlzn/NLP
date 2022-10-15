import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from dataset import text_CLS, data_loader
from configs import config

if __name__ == '__main__':
    cfg = config()
    data_path = '../data/weibo_senti_100k.csv'
    data_stop_path = '../data/hit_stopword'
    dict_path = '../data/dict'
    dataset = text_CLS(dict_path, data_path, data_stop_path)
    cfg.pad_size = dataset.max_len_seq
    train_loader = data_loader(dataset, cfg)
    model_text_cls = Model(config=cfg)
    model_text_cls.load_state_dict(torch.load('./data/model_9.pth'))
    model_text_cls.to(cfg.devices)

    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        label, data = batch
        label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)
        data = torch.tensor(data).to(cfg.devices)
        out = model_text_cls(data)
        pred = torch.argmax(out, dim=1)
        out = torch.eq(pred, label)
        print(out.sum() * 1.0 / pred.size()[0])
        # pred = torch.argmax(out)
        # print(pred, label)
