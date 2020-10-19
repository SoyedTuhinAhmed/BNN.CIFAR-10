import torch.nn as nn


class LeNet5Cifar10(nn.Module):
    def __init__(self):
        super(LeNet5Cifar10, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # out_dim= 32 * 32 * 32
            nn.Hardtanh(),
            nn.MaxPool2d(2, 2),  # out_dim= 16 * 16 * 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # out_dim= 16 * 16 * 64
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # out_dim= 8 * 8 * 64
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 64, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape) # shape of x helps choose the size of 1st linear layer and flatten dimensions
        x = x.view(-1, 8 * 8 * 64)
        x = self.classifier(x)
        return x
