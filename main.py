# import all torch related modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import TensorDataset 
# add mnist dataset
from torchvision import datasets, transforms
import pandas as pd
import numpy as np


# get the train.csv file
train = pd.read_csv('./train.csv')

# get 4 faeutures and 1 label
features = train.iloc[:, 1:].values
# get 1 3 4  17 19 56 features 
features = train.iloc[:, [1, 3, 4, 17, 19, 56]].values
# get the label
labels = train.iloc[:, -1].values

# convert the features and labels to long tensor
features = torch.from_numpy(features).long()
labels = torch.from_numpy(labels).long()




# get mnist dataset
# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

class Model:
    def __init__(self, model, optimizer, train_loader):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
    def train(self, epoch):
        train(self.model, self.train_loader, self.optimizer, epoch)

# flatten the image
def flatten_image(image):
    return image.view(image.size(0), -1)

def feed_forward():
    # define a feed forward neural network
    class FeedForward(nn.Module):
        def __init__(self):
            super(FeedForward, self).__init__()
            self.fc1 = nn.Linear(6, 10)
            self.fc2 = nn.Linear(10, 5)
            self.fc3 = nn.Linear(5, 1)
        def forward(self, x):
            # x = x.view(-1, 6)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    return FeedForward()


# train the model
def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def run():
    # define a feed forward neural network
    model = feed_forward()
    # define an optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # define a train loader
    train_loader = DataLoader(TensorDataset(features, labels), batch_size=64, shuffle=True)
    # train the model
    for epoch in range(1, 10 + 1):
        train(model, train_loader, optimizer, epoch)


if __name__ == '__main__':
    run()