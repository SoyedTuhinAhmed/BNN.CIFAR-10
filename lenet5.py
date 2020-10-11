import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, humult):
        super(LeNet5, self).__init__()
        self.infl_ratio = humult  # Hidden unit multiplier, how many hidden unit yot waht to have

        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # binarize the convolution layer
        self.conv1_bn = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # binarize the convolution layer
        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16*5*5, int(120 * self.infl_ratio))
        self.fc1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(int(120 * self.infl_ratio), int(84 * self.infl_ratio))
        self.fc1 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(int(84 * self.infl_ratio), 10)
        self.htanh1 = nn.Hardtanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = nn.MaxPool2d(out, 2)
        out = nn.hardtanh(out)

        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = nn.MaxPool2d(out, 2)
        out = F.Ha
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.hardtanh(self.fc1(out))
        out = F.hardtanh(self.fc2(out))
        out = self.fc3(out)
        return out
