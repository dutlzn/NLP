import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from lstm import Model
from dataset import text_CLS, data_loader
from configs import config

if __name__ == '__main__':
    cfg = config()
    data_path = '../data/weibo_senti_100k.csv'
    data_stop_path = '../data/hit_stopword'
    dict_path = '../data/dict'
    dataset = text_CLS(cfg, dict_path, data_path, data_stop_path)
    cfg.pad_size = dataset.max_seq_len
    cfg.max_seq_len = dataset.max_seq_len
    train_loader = data_loader(dataset, cfg)
    model_text_cls = Transformer(config=cfg)
    # model_text_cls = Model(cfg)
    model_text_cls.load_state_dict(torch.load('../data/mode.pth'))
    model_text_cls.to(cfg.devices)


    sum = 0
    for i, batch in enumerate(train_loader):
        label, data = batch
        label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)
        data = torch.tensor(data).to(cfg.devices)
        input_pad = 1001
        input_mask = (data != input_pad)
        out = model_text_cls(data, input_mask)
        # out = model_text_cls(data)
        pred = torch.argmax(out, dim=1)
        out = torch.eq(pred, label)
        sum += out.sum() * 1.0
    print(sum / len(train_loader.dataset))
        # pred = torch.argmax(out)