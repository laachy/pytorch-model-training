import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, output_size, width=256, depth=3, funnel_factor=1, dropout=0.3, lr=4e-4, weight_decay=3e-2, min_width=64):
        super().__init__()

        w = width   # set intial
        # create network
        layers = [nn.Flatten()]
        for i in range(depth):
            layers += [nn.LazyLinear(w), nn.ReLU(), nn.Dropout(dropout)]
            w = max(w//funnel_factor, min_width)  # update width for next iter 
        layers += [nn.LazyLinear(output_size)]

        self.net = nn.Sequential(*layers)

        # other hparams
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def configure_optimisers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @classmethod
    def build_for_experiment(cls, output_size, trial=None):
        if not trial:
            return cls(output_size)

        depth = trial.suggest_int("depth", 1, 6)
        width = trial.suggest_categorical("width", [64, 128, 256, 512])
        funnel_factor = trial.suggest_categorical("funnel_factor", [1, 2, 3])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    
        return cls(output_size, width, depth, funnel_factor, dropout, lr, weight_decay)
 
        
