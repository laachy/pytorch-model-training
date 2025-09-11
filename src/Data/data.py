import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DataModule:
    def __init__(self, train_ds=None, val_ds=None, train_root=None, val_root=None, num_workers=4):
        self.num_workers = num_workers
        self.pin = torch.cuda.is_available()

        if train_ds and val_ds:
            self.train_ds = train_ds
            self.val_ds = val_ds
        elif train_root and val_root:
            self.setup(train_root, val_root)

    def _transforms(self):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((-1.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # setup using an on disk path
    def setup(self, train_root, val_root):
        transform = self._transforms()
        self.train_ds = datasets.ImageFolder(root=train_root, transform=transform)
        self.val_ds = datasets.ImageFolder(root=val_root, transform=transform)
        
    def train_dataloader(self, batch_size):
        return DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin)

    def val_dataloader(self, batch_size):
        return DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin)

    def output_size(self):
        return len(self.train_ds.classes)
