import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataModule:
    def __init__(self, transform, train_ds=None, val_ds=None, test_ds=None, train_root=None, val_root=None, test_root=None, num_workers=4):
        self.num_workers = num_workers
        self.pin = torch.cuda.is_available()
        self.tfm = transform

        # TODO: CHANGE TO CHECK INDIVIDUALLY TO ALLOW MIXED METHODS OF DATA DEFINITION AND NOT REQUIRE TEST
        if train_ds and val_ds and test_ds:
            self.train_ds = train_ds
            self.val_ds = val_ds
            self.test_ds = test_ds
        elif train_root and val_root and test_root:
            self.setup(train_root, val_root, test_root, transform)

    # setup using an on disk path
    def setup(self, train_root, val_root, test_root, transform):
        self.train_ds = datasets.ImageFolder(root=train_root, transform=transform)
        self.val_ds = datasets.ImageFolder(root=val_root, transform=transform)
        self.test_ds = datasets.ImageFolder(root=test_root, transform=transform)
        
    def train_dataloader(self, batch_size):
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin)

    def val_dataloader(self, batch_size):
        return DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin)

    def test_dataloader(self, batch_size):
        return DataLoader(self.test_ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin)

    def classes(self):
        return self.train_ds.classes

    @classmethod
    def find_normalise(cls, tfm):
        norms = []
        if hasattr(tfm, "transforms"): 
            for t in tfm.transforms:
                norms.extend(cls.find_normalise(t))
        if hasattr(tfm, "mean") and hasattr(tfm, "std"):
            norms.append(tfm)

        return norms

    @classmethod
    def denorm(cls, tfm, input):
        norms = cls.find_normalise(tfm)

        mean = torch.tensor(norms[-1].mean)
        std = torch.tensor(norms[-1].std)

        if input.dim() == 4:      # [B,C,H,W]
            m = mean.view(1, -1, 1, 1)
            s = std.view(1, -1, 1, 1)
        elif input.dim() == 3:    # [C,H,W]
            m = mean.view(-1, 1, 1)
            s = std.view(-1, 1, 1)

        return (input * s + m).clamp(0, 1)
