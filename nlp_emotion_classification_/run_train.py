import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from dataset import text_CLS, data_loader
from configs import config

if __name__ == '__main__':
    cfg = config()
    data_path = './data/weibo_senti_100k.csv'
    data_stop_path = './data/hit_stopword'
    dict_path = './data/dict'
    dataset = text_CLS(dict_path, data_path, data_stop_path)
    cfg.pad_size = dataset.max_len_seq
    train_loader = data_loader(dataset, cfg)
    model_text_cls = Model(config=cfg)
    model_text_cls.to(cfg.devices)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.learning_rate)


    for epoch in range(cfg.num_epochs):
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            label, data = batch
            label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)
            data = torch.tensor(data).to(cfg.devices)

            optimizer.zero_grad()
            out = model_text_cls(data)
            loss = loss_fun(out, label)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        print('epoch: {}, train_loss:{}'.format(epoch + 1, train_loss / len(train_loader.dataset)))

        torch.save(model_text_cls.state_dict(), "./data/model_{}.pth".format(epoch))