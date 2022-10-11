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



# get mnist dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

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
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    return FeedForward()


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

#run the neural network
def run():
    model = feed_forward()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    train_loader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    model = Model(model, optimizer, train_loader)
    for epoch in range(1, 10 + 1):
        model.train(epoch)


if __name__ == '__main__':
    run()