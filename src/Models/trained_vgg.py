import torch
from torchvision.models import vgg16, VGG16_Weights

# https://medium.com/@piyushkashyap045/transfer-learning-in-pytorch-fine-tuning-pretrained-models-for-custom-datasets-6737b03d6fa2
class TrainedVGG(torch.nn.Module):
    weights = VGG16_Weights.DEFAULT     # should be best weights

    def __init__(self, output_size, lr=1e-3, weight_decay=0.5):
        super().__init__()
        self.net = vgg16(weights=self.weights)

        # change output layer
        self.net.classifier[-1] = torch.nn.LazyLinear(output_size)

        # Freeze all layers
        for p in self.net.parameters():
            p.requires_grad = False

        # unfreeze classifier layers
        for p in self.net.classifier.parameters():
            p.requires_grad = True

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        return self.loss_fn(y_hat, y)

    def configure_optimisers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    @classmethod
    def transforms(cls):
        return cls.weights.transforms()

    @classmethod
    def build_for_experiment(cls, output_size, trial=None):
        if not trial:
            batch_size = 32     # default
            return cls(output_size), batch_size

        batch_size = trial.suggest_int("batch_size", 16, 64, step=16)

        lr = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
    
        return cls(output_size, lr, weight_decay), batch_size