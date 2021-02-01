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
gen = generator.Generator(100, 784)
dis = discriminator.Discriminator(784, 1)

discriminator_steps = 10
generator_steps = 10

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


# training

for epoch in range(epochs):
    
    # training discriminator
    for d_step in range(discriminator_steps):
        dis.zero_grad()

        # on real data
        for inp_real, _ in dataloader_mnist_train:
            inp_real_x = inp_real
            break
        
        inp_real_x = inp_real_x.reshape(4, 784)
        dis_real_out = dis(inp_real_x)
        dis_real_loss = criterion_d1(dis_real_loss, Variable(torch.ones(batch_size, 1)))
        dis_real_loss.backward()

        # on generated data
        inp_fake_x_gen = make_some_noise()
        dis_inp_fake_x = gen(inp_fake_x_gen.detach())
        dis_fake_out = dis(dis_inp_fake_x)
        dis_fake_loss = criterion_d1(dis_fake_out, Variable(torch.zeros(batch_size, 1)))
        dis_fake_loss.backward()

        optimizer_d1.step()
    
    # training generator
    for g_step in range(generator_steps):
        gen.zero_grad()

        # generating data for input for generator
        gen_inp = make_some_noise()

        gen_out = gen(gen_inp)
        dis_out_gen_training = dis(gen_out)
        gen_loss = criterion_d2(dis_out_gen_training, Variable(torch.ones(batch_size, 1)))
        gen_loss.backward()

        optimizer_d2.step()

    if epoch % printing_steps == 0:
        print("epoch no. {}".format(epoch))
        plot_image(gen_out[0])
        plot_image(gen_out[1])
        plot_image(gen_out[2])
        plot_image(gen_out[3])
        print("--- END OF EPOCH ---")

