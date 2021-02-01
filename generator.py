import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms

# class for the generator NN
class Generator(nn.Module):
	def __init__(self, inp, out):
		super().__init__()
		self.net = nn.Sequential(nn.Linear(inp, 300), 
								 nn.ReLU(inplace=True),
								 nn.Linear(300, 1000),
								 nn.ReLU(inplace=True),
								 nn.Linear(1000, 800),
								 nn.ReLU(inplace=True),
								 nn.Linear(800, out))
	def forward_pass(self, x):
		x = self.net(x)
		return x