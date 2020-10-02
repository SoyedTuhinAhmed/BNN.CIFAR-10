import torch.nn as nn
from argument_parser import *


class CovNet(nn.Module):
    def __init__(self):
        super(CovNet, self).__init__()
        arg_pass = ArgumentParser()
        self.infl_ratio = arg_pass.args.humult  # Hidden unit multiplier, how many hidden unit yot waht to have
        # 28x28 = 784 is the input layer and multiple of 128 is the hidden unit
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv1 = BinarizeConv2d(3, 6, kernel_size=5, stride=1, padding=0,
                                    bias=True)  # binarize the convolution layer
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2 = BinarizeConv2d(3, 6, kernel_size=5, stride=1, padding=0,
                                    bias=True)  # binarize the convolution layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = BinarizeLinear(784, int(128 * self.infl_ratio))
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc1 = BinarizeLinear(784, int(128 * self.infl_ratio))  # binarize the layer
        self.htanh1 = nn.Hardtanh()  # apply hyperbolic tangent function
        self.fc2 = nn.Linear(int(128 * self.infl_ratio), 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        return x
