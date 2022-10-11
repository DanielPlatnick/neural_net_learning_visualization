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
        self.fc1 = nn.Linear(6, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        # add output layer
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        # add 2 hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # add output layer
        x = self.fc4(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def run():
    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create model
    model = Model().to(device)
    # create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # create dataset
    dataset = TensorDataset(features, labels)
    # split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    # train model
    for epoch in range(1, 10 + 1):
        train(model, device, train_loader, optimizer, epoch)
    # test model
    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
    #         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)
    
    # print("""
    # Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)
    # """.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    

if __name__ == '__main__':
    run()