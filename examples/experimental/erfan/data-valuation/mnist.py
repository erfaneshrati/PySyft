from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class MNIST:
    def __init__(self):
        transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])

        self.train_set = datasets.MNIST('./data', train=True, download=True,
                             transform=transform)
        self.test_set = datasets.MNIST('./data', train=False,
                            transform=transform)

        use_cuda = torch.cuda.is_available()
        torch.manual_seed(0)
        device = torch.device("cuda" if use_cuda else "cpu")
        train_kwargs = {'batch_size': 128}
        test_kwargs = {'batch_size': 128}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        train_loader = torch.utils.data.DataLoader(self.train_set,**train_kwargs)
        test_loader = torch.utils.data.DataLoader(self.test_set, **test_kwargs)
        self.model = Net().to(device)
        optimizer = optim.Adadelta(self.model.parameters(), lr=0.001)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
        checkpoint = "mnist_cnn.pt"
        if os.path.exists(checkpoint):
            self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        else:
            for epoch in range(1, 10):
                self.train(self.model, device, train_loader, optimizer, checkpoint, epoch)
                scheduler.step()
        embed_path = 'mnist_test_embedds.npy'
        if not os.path.exists(embed_path):
            self.test(self.model, device, test_loader)
        self.test_embeds = np.load(embed_path)

    def train(self, model, device, train_loader, optimizer, checkpoint, epoch=10, log_interval=100):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            _, output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        torch.save(model.state_dict(), checkpoint)

    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        batch_size = test_loader.batch_size
        embeddings = np.zeros((len(test_loader.dataset), 128+1)) #128 is for embedding, 1 is for label
        labels = np.zeros((len(test_loader.dataset)))
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                embedding, output = model(data)
                embeddings[i*batch_size:i*batch_size+len(data), 0:128] = embedding.cpu().numpy()
                embeddings[i*batch_size:i*batch_size+len(data), 128] = target.cpu().numpy()
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        np.save('mnist_test_embedds', embeddings)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        embedding = F.relu(x)
        x = self.dropout2(embedding)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return embedding, output

if __name__ == '__main__':
    dataset = MNIST()

