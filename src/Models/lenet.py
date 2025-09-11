import torch
from torch import nn

# https://classic.d2l.ai/chapter_convolutional-neural-networks/lenet.htmls
class LeNet(nn.Module):
    def __init__(self, output_size, lr=0.01):
        super().__init__()

        self.net= nn.Sequential(
            nn.LazyConv2d(out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(out_channels=16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, output_size)
        )

        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def configure_optimisers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    @classmethod
    def build_for_experiment(cls, output_size, trial=None):
        if not trial:
            return cls(output_size)
