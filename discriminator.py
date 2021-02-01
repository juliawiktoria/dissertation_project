import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms

# class for the discriminator NN
class Discriminator(nn.Module):
    def __init__(self, inp, out):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(nn.Linear(inp, 300), 
								 nn.ReLU(inplace=True),
								 nn.Linear(300, 300),
								 nn.ReLU(inplace=True),
								 nn.Linear(300, 200),
								 nn.ReLU(inplace=True),
								 nn.Linear(200, out),
                                 nn.Sigmoid())
    def forward_pass(self, x):
        x = self.net(x)
        return x
