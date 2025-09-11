import torch
from torch import nn

# https://arxiv.org/pdf/1409.1556
# https://www.digitalocean.com/community/tutorials/vgg-from-scratch-pytorch#vgg16-from-scratch
class VGG(nn.Module):
    #ARCH = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))    # VGG16 arch
    ARCH = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))    # VGG11

    def __init__(self, output_size, lr=1e-3, weight_decay=5e-4):
        super().__init__()

        layers = []
        # conv part
        for (num_convs, out_channels) in self.ARCH:
            layers += self.block(num_convs, out_channels)

        # fully connected part
        layers += [
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, output_size)
        ]
        self.net = nn.Sequential(*layers)

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = nn.CrossEntropyLoss()

    def block(self, num_convs, out_channels):
        layers = []
        for _ in range(num_convs):
            layers += [nn.LazyConv2d(out_channels, kernel_size=3, padding=1), 
                nn.BatchNorm2d(out_channels),
                nn.ReLU()]

        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return layers

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

