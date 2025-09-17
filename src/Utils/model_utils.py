
import torch.nn as nn
import torch
def str_to_activation(name):
    match name:
        case 'relu':
            activation_fn = nn.ReLU()
        case 'sigmoid':
            activation_fn = nn.Sigmoid()
        case 'tanh':
            activation_fn = nn.Tanh()

    return activation_fn

def str_to_optimiser(name):
    match name:
        case 'Adam':
            optimiser = torch.optim.Adam
        case 'AdamW':
            optimiser = torch.optim.AdamW
        case 'SGD':
            optimiser = torch.optim.SGD
        
    return optimiser