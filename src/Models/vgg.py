import torch
from torch import nn
from torchvision import transforms

# https://arxiv.org/pdf/1409.1556
# https://www.digitalocean.com/community/tutorials/vgg-from-scratch-pytorch#vgg16-from-scratch
class VGG(nn.Module):
    ARCH = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))    # VGG16 arch
    #ARCH = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))    # VGG11

    def __init__(self, output_size, fc_depth=4096, dropout=0.5, lr=1e-3, weight_decay=5e-4, batch_norm=True):
        super().__init__()

        in_channels = 3
        layers = []
        # conv part
        for (num_convs, out_channels) in self.ARCH:
            layers += self.block(num_convs, in_channels, out_channels, batch_norm)
            in_channels = out_channels

        # fully connected part
        layers += [
            nn.Flatten(),
            nn.Linear(out_channels*7*7, fc_depth), nn.ReLU(), nn.Dropout(dropout),
            nn.LazyLinear(fc_depth), nn.ReLU(), nn.Dropout(dropout),
            nn.LazyLinear(output_size)
        ]
        self.net = nn.Sequential(*layers)

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()

    def block(self, num_convs, in_channels, out_channels, batch_norm):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels

        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return layers

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def configure_optimisers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

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
            batch_size = 32     # default
            return cls(output_size), batch_size

        batch_size = trial.suggest_int("batch_size", 16, 64, step=16)
        
        fc_width = trial.suggest_categorical("fc_width", [512, 1024, 2048, 4096])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    
        return cls(output_size, fc_width, dropout, lr, weight_decay), batch_size

