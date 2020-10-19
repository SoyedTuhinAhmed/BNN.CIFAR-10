import torch.nn as nn
from binarized_modules import BinarizeLinear, BinarizeConv2d


class BinarizedLeNet5Cifar10(nn.Module):
    def __init__(self):
        super(BinarizedLeNet5Cifar10, self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(3, 32, kernel_size=3, padding=1),  # out_dim= 32 * 32 * 32
            nn.MaxPool2d(2, 2),  # out_dim= 16 * 16 * 32
            nn.BatchNorm2d(32),
            nn.Hardtanh(),

            BinarizeConv2d(32, 64, kernel_size=3, padding=1),  # out_dim= 16 * 16 * 64
            nn.MaxPool2d(2, 2),  # out_dim= 8 * 8 * 64
            nn.BatchNorm2d(64),
            nn.Hardtanh()
        )

        self.classifier = nn.Sequential(
            BinarizeLinear(8 * 8 * 64, 512),
            nn.BatchNorm1d(512),
            nn.Hardtanh(),

            BinarizeLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.Hardtanh(),

            nn.Linear(256, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape) # shape of x helps choose the size of 1st linear layer and flatten dimensions
        x = x.view(-1, 8 * 8 * 64)
        x = self.classifier(x)
        return x
