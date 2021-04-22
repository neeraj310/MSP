# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# The Beta Finder Network
class BFNet(nn.Module):
    def __init__(self, classes=2):
        super(BFNet, self).__init__()
        self.classes = 2
        # Convolution
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(10,2),stride=(5,1))
        self.relu1 = nn.ReLU(inplace=True)

        # Fully-connected for classification
        self.fc1 = nn.Conv2d(10, classes, (3,1))

        # ConvTranspose
        self.upscore1 = nn.ConvTranspose2d(classes, 10, kernel_size=(3,1))
        self.upscore2 = nn.ConvTranspose2d(10,
                                           1,
                                           kernel_size=(10,1),
                                           stride=(5,1),
                                           bias=True)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.relu1(h)

        h = self.fc1(h)
        h = self.upscore1(h)
        h = self.upscore2(h)
        return h


class BFModel(object):
    def __init__(self, num_breaks) -> None:
        super().__init__()
        self.num_breaks = num_breaks
        self.net = BFNet()
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.0001, momentum=0.9)

    def train(self, X, Y):
        X = X.reshape((1, 1, X.shape[0], X.shape[1]))
        Y = Y.reshape((1, 1, Y.shape[0], 1))
        X = torch.Tensor(X)
        Y = torch.Tensor(Y)
        train_data = TensorDataset(X, Y)
        dataloader = DataLoader(train_data, batch_size=1)
        for epoch in range(30):
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                ypred = self.net.forward(inputs)
                loss = self.loss(ypred, labels)
                loss.backward()
                self.optimizer.step()
                print("epoch: {} Loss:{}".format(epoch, loss),end="\n")
        print("Finished trainning...")
        torch.save(self.net.state_dict(), './bfnet.model')

    def load(self):
        self.net.load_state_dict(torch.load('./bfnet.model'))

    def predict(self, X):
        X = X.reshape((1, 1, X.shape[0], X.shape[1]))
        X = torch.Tensor(X)
        output = self.net.forward(X)
        return output.detach().numpy()