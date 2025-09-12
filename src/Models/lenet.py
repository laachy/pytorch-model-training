import torch
from torch import nn
from torchvision import transforms

# https://classic.d2l.ai/chapter_convolutional-neural-networks/lenet.htmls
class LeNet(nn.Module):
    def __init__(self, output_size, lr=0.01, weight_decay=5e-4):
        super().__init__()

        in_channels = 1

        self.net= nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, output_size)
        )

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
    def transforms(cls):
        mean, std = (0.5), (0.5)
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28,28)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    @classmethod
    def build_for_experiment(cls, output_size, trial=None):
        if not trial:
            batch_size = 64
            return cls(output_size), batch_size

        batch_size = trial.suggest_int("batch_size", 32, 128, step=32)

        lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)

        return cls(output_size, lr, weight_decay), batch_size

