import torch
from torch import nn
from torchvision import transforms
from Utils.model_utils import str_to_activation, str_to_optimiser

# https://arxiv.org/pdf/1409.1556
# https://www.digitalocean.com/community/tutorials/vgg-from-scratch-pytorch#vgg16-from-scratch
class VGG(nn.Module):
    ARCH = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))    # VGG16 arch
    #ARCH = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))    # VGG11

    def __init__(self, output_size, fc_width=4096, dropout=0.5, lr=1e-3, weight_decay=5e-4, activation_fn=nn.ReLU, optimiser=torch.optim.AdamW, batch_norm=True):
        super().__init__()

        in_channels = 3
        layers = []
        # conv part
        for (num_convs, out_channels) in self.ARCH:
            layers += self.block(num_convs, in_channels, out_channels, activation_fn, batch_norm)
            in_channels = out_channels

        # fully connected part
        layers += [
            nn.Flatten(),
            nn.Linear(out_channels*7*7, fc_width), activation_fn, nn.Dropout(dropout),
            nn.LazyLinear(fc_width), activation_fn, nn.Dropout(dropout),
            nn.LazyLinear(output_size)
        ]
        self.net = nn.Sequential(*layers)

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimiser = optimiser

    def block(self, num_convs, in_channels, out_channels, activation_fn, batch_norm):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_fn)
            in_channels = out_channels

        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return layers

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
            batch_size = 16     # default
            return cls(output_size), batch_size

        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        fc_width = trial.suggest_categorical("fc_width", [4096])    # can change later if needed but stay to original arch

        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-7, 1e-1, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-8, 1e-2, log=True)

        activation_fn = str_to_activation(trial.suggest_categorical('activation_fn', ['relu', 'sigmoid', 'tanh']))
        optimiser = str_to_optimiser(trial.suggest_categorical("optimiser", ["Adam", "AdamW", "SGD"]))
    
        return cls(output_size, fc_width, dropout, lr, weight_decay, activation_fn, optimiser), batch_size

