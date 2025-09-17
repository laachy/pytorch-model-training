from torchvision.datasets import CIFAR10, MNIST
import os, torch

from Models import *
from Training.experiment import Experiment, Study
from Data.data import DataModule

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import *


def main():
    model_cls = VGG

    model_name = input("enter new or existing model: ")
    model_path = f"{MODEL_DIR}/{model_name}"

    # Loading the data
    #train_ds = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    #val_ds = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
    #dm = DataModule(train_ds=train_ds, val_ds=val_ds)
    dm = DataModule(model_cls.transforms(), num_workers=4, train_root=TRAIN_ROOT, val_root=VAL_ROOT, test_root=TEST_ROOT)

    experiment = Experiment(dm, model_cls, model_path, MAX_EPOCHS)
    # training (if new model name with no ckpt)
    if not os.path.isfile(f"{model_path}/{CKPT_NAME}"):
        Study(model_name).optimise(experiment, n_trials=N_TRIALS)

    # testing
    experiment.test()

if __name__ == "__main__":
    main()

    

    

    
