import torch
import torch.nn as nn
import torch.optim as optim
from transformer import Transformer
from rnn import Rnn
from configs import config
import warnings
import random
import numpy as np

from datasets import TextDatasets

from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')

# 设置随机种子
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    setup_seed(1)
    cfg = config()
    data_path = './data/weibo_senti_100k.csv'
    data_stop_path = './data/hit_stopword'
    dict_path = './data/dict'
    dataset = TextDatasets(data_path, data_stop_path, dict_path, cfg)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    # model_text_cls = Transformer(cfg)
    model_text_cls = Rnn(cfg)
    # 开启训练
    model_text_cls.load_state_dict(torch.load('../data/rnn.pth'))
    model_text_cls.to(cfg.devices)

    model_text_cls.eval()
    sum = 0
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        data, label = batch
        label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)
        data = torch.tensor(data, dtype=torch.int64).to(cfg.devices)

        input_pad = 1001
        input_mask = (data != input_pad)

        # out, _, _ = model_text_cls(data, input_mask)
        out =  model_text_cls(data)
        pred = out.argmax(dim=1)
        out = torch.eq(pred, label).float().sum().item()
        sum += out


    print(sum / len(train_loader.dataset))