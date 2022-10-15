import torch
class config():
    def __init__(self):
        self.hidden_size = 128
        self.num_layers = 3
        self.pad_size = 400

        self.num_classes = 2
        self.num_epochs = 10
        self.d_ff = 256
        self.embed_size = 128
        self.n_vocab = 1002
        self.learning_rate = 0.001
        self.dropout = 0.8
        self.batch_size = 8 * 8 * 2
        self.shuffle = True
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_seq_len = 60