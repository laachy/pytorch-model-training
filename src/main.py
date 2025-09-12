from torchvision.datasets import CIFAR10, MNIST
import os, torch

from Models import *
from Training.experiment import Experiment, Study
from Data.data import DataModule

# location of data
TRAIN_ROOT = "../data/train"
VAL_ROOT = "../data/valid"

MODEL_NAME = "vgg16"
MODEL_DIR = "../model_data"

MAX_EPOCHS = 20
N_TRIALS = 100

def main():
    model = MLP
    transform = model.transforms()

    # Loading the data
    #train_ds = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    #val_ds = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)
    #dm = DataModule(train_ds=train_ds, val_ds=val_ds)
    dm = DataModule(transform, train_root=TRAIN_ROOT, val_root=VAL_ROOT)

    model_path = f"{MODEL_DIR}/{MODEL_NAME}"
    experiment = Experiment(dm, model, model_path, MAX_EPOCHS)
    #experiment.run()

    Study(MODEL_NAME).optimise(experiment, n_trials=N_TRIALS)

if __name__ == "__main__":
    main()

    

    

    
