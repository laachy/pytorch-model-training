import torch
from torch import nn
from torchvision import transforms
from Utils.model_utils import str_to_activation, str_to_optimiser

class MLP(nn.Module):
    def __init__(self, output_size, widths=[512, 256], dropout=0.3, lr=4e-4, weight_decay=3e-2, activation_fn=nn.ReLU, optimiser=torch.optim.AdamW):
        super().__init__()

        # create network
        layers = [nn.Flatten()]
        for w in widths:
            layers += [nn.LazyLinear(w), activation_fn, nn.Dropout(dropout)]
        layers += [nn.LazyLinear(output_size)]

        self.net = nn.Sequential(*layers)

        # other hparams
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimiser = optimiser

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def configure_optimisers(self):
        return self.optimiser(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    @classmethod
    def transforms(cls):
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    @classmethod
    def build_for_experiment(cls, output_size, trial=None):
        if not trial:
            batch_size = 64     # default
            return cls(output_size), batch_size

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])

        depth = trial.suggest_int("depth", 1, 6)
        widths = [trial.suggest_categorical(f"width_{i}", [64, 128, 256, 512, 1024, 2048]) for i in range (depth)]

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)

        activation_fn = str_to_activation(trial.suggest_categorical('activation_fn', ['relu', 'sigmoid', 'tanh']))
        optimiser = str_to_optimiser(trial.suggest_categorical("optimiser", ["Adam", "AdamW", "SGD"]))
    
        return cls(output_size, widths, dropout, lr, weight_decay, activation_fn, optimiser), batch_size

    @classmethod
    def extract(cls, hp):
        hp2 = {k: v for k, v in hp.items() if k != "depth" and not k.startswith("width_")}  # rebuild hp without depth and w

        widths = []
        for k in hp:
            if k.startswith("width_"):
                widths.append(hp[k])

        hp2["widths"] = widths
        return hp2