import torch
import torch.nn as nn

# class for the generator NN
class Generator(nn.Module):
	def __init__(self, inp, out):
		super(Generator, self).__init__()

		self.net = nn.Sequential(nn.Linear(inp, 300), 
								 nn.ReLU(inplace=True),
								 nn.Linear(300, 1000),
								 nn.ReLU(inplace=True),
								 nn.Linear(1000, 800),
								 nn.ReLU(inplace=True),
								 nn.Linear(800, out))
	def forward(self, x):
		x = self.net(x)
		return x