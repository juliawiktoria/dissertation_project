import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms

# main body of the program, setting parameters and args

# noise generating function
def make_some_noise():
    return torch.rand(batch_size,100)

# data from MNIST dataset
data_transforms = transforms.Compose([transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transforms)
batch_size=4
dataloader_mnist_train = t_data.DataLoader(mnist_trainset, 
                                           batch_size=batch_size,
                                           shuffle=True)