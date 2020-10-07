import torch.nn as nn
from argument_parser import *
from models.binarized_modules import *


class BinaryLeNet5(nn.Module):
    def __init__(self, arg_pass):
        super(BinaryLeNet5, self).__init__()
        self.arg_pass = arg_pass
        self.infl_ratio = self.arg_pass.args.humult  # Hidden unit multiplier, how many hidden unit yot waht to have

        self.conv1 = BinarizeConv2d(3, 6, kernel_size=5, stride=1, padding=0,
                                    bias=True)  # binarize the convolution layer

        self.conv2 = BinarizeConv2d(3, 6, kernel_size=5, stride=1, padding=0,
                                    bias=True)  # binarize the convolution layer

        self.fc1 = BinarizeLinear(784, int(120 * self.infl_ratio))
        self.fc2 = BinarizeLinear(int(120 * self.infl_ratio), int(84 * self.infl_ratio))
        self.fc3 = BinarizeLinear(int(84 * self.infl_ratio), 10)
        self.htanh1 = nn.Hardtanh()

    def forward(self, x):
        out = nn.Hardtanh(self.conv1(x))
        out = nn.max_pool2d(out, 2)
        out = nn.Hardtanh(self.conv2(out))
        out = nn.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = nn.Hardtanh(self.fc1(out))
        out = nn.Hardtanh(self.fc2(out))
        out = self.fc3(out)
        return out

        # 28x28 = 784 is the input layer and multiple of 128 is the hidden unit
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = BinarizeLinear(784, int(128 * self.infl_ratio))  # binarize the layer
        # self.htanh1 = nn.Hardtanh()  # apply hyperbolic tangent function
        # self.fc2 = nn.Linear(int(128 * self.infl_ratio), 10)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)

        # x = x.view(-1, 28 * 28)
        # x = self.fc1(x)
        # x = self.htanh1(x)
        # x = self.fc2(x)