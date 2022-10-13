import torch.cuda


class config():
    """
     self.embedding = nn.Embedding(config.n_vocab,
                                      config.embed_size,
                                      padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed_size,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=True)

        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size*2 + config.embed_size,
                            config.num_classes)
    """
    def __init__(self):
        self.n_vocab = 1002
        self.embed_size = 128
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout = 0.8
        self.pad_size = 32
        self.num_classes = 2
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 128
        self.shuffle = True
        self.learning_rate = 0.001
        self.num_epochs = 10


