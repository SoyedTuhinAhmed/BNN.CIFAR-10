import torch.nn as nn


class Cifar10MLP(nn.Module):
    def __init__(self):
        super(Cifar10MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * 32 * 32, 2048),
            nn.ReLU(),

            nn.Linear(2048, 2048),
            nn.ReLU(),

            nn.Linear(2048, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),

            nn.Linear(2048, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # convert 1*28*28 input tensor to 28*28 vector
        x = self.fc(x)
        return x
