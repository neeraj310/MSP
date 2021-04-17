# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import torch.nn as nn


class BFNet(nn.Module):
    def __init__(self, num_breakpoints, classes=2):
        super(BFNet, self).__init__()
        self.classes = 2
        # Convolution
        self.conv1 = nn.Conv2d(1, 10, 3)
        self.relu1 = nn.ReLU(inplace=True)

        # Fully-connected for classification
        self.fc1 = nn.Conv2d(10, classes, 1)

        # ConvTranspose
        self.upscore1 = nn.ConvTranspose2d(classes,
                                           classes,
                                           stride=2,
                                           bias=False)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.relu1(h)

        h = self.fc1(h)
        h = self.upscore1(h)
        return h


class BFModel(object):
    def __init__(self, num_breaks) -> None:
        super().__init__()
        self.num_breaks = num_breaks
        self.net = BFNet(num_breaks)

    def train(self, X, breaks):
        for epoch in range(500):
            for i in X:
                pass
