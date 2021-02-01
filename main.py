import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as t_data
import torchvision.datasets as datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import generator
import discriminator

# main body of the program, setting parameters and args

# initialise generator and discriminator instances
gen = generator.Generator(100, 100)
dis = discriminator.Discriminator(100, 100)

discriminator_steps = 100
generator_steps = 100

# discriminator
criterion_d1 = nn.BCELoss()
optimizer_d1 = optim.SGD(dis.parameters(), lr=0.001, momentum=0.9)

# generator
criterion_d2 = nn.BCELoss()
optimizer_d2 = optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)

printing_steps = 200

epochs = 50

# noise generating function
def make_some_noise():
    return torch.rand(batch_size,100)

def plot_image(array, number=None):
    array = array.detach()
    array = array.reshape(28, 28)

    plt.imshow(array, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    if number:
        plt.xlabel(number, fontsize='x-large')
    plt.show()

# data from MNIST dataset
data_transforms = transforms.Compose([transforms.ToTensor()])
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=data_transforms)
batch_size=4
dataloader_mnist_train = t_data.DataLoader(mnist_trainset, 
                                           batch_size=batch_size,
                                           shuffle=True)