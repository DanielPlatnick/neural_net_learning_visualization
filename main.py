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

# get 1 3 4  17 19 56 features 


# drop feautres with 0 values
train = train.loc[(train!=0).any(axis=1)]
# fill na values with 0
train = train.fillna(0)

# drop features with constant values
train = train.loc[:, (train != train.iloc[0]).any()]
features = train.iloc[:, [1, 3, 4, 17, 19, 56]].values




# get the label
labels = train.iloc[:, -1].values

# convert the features to tensor
features = torch.from_numpy(features).float()
# convert the labels to tensor
labels = torch.from_numpy(labels).float()

# reshape labels to (batch_size, 1)
labels = labels.view(-1, 1)





class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # add 2 hidden layers
        self.fc1 = nn.Linear(6, 100)
        self.fc2 = nn.Linear(100, 10)
        # add output layer
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        # sigmoid activation function
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        # clip gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def run():
    # use GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # create model
    model = Model()
    # move model to GPU
    model.to(device)
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # create dataset
    dataset = TensorDataset(features, labels)
    # create data loader
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # train the model
    for epoch in range(1, 10 + 1):
        train(model, device, train_loader, optimizer, epoch)
    # save the model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    # run()
    import tensorflow as tf
