import torch

class config:
    def __init__(self):
        self.num_epochs = 10
        self.learning_rate = 0.001
        self.n_vocab = 1002
        self.embed_size = 128
        self.dropout = 0.8
        self.max_seq_len = 60
        self.d_ff = 1024
        self.batch_size = 64 * 2
        self.shuffle = True
        self.devices = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 128
        self.layer_num = 3
        self.num_classes = 2
