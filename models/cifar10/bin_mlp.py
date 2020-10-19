import torch.nn as nn
from binarized_modules import BinarizeLinear


class BinarizedCifar10MLP(nn.Module):
    def __init__(self):
        super(BinarizedCifar10MLP, self).__init__()
        self.fc = nn.Sequential(
            BinarizeLinear(3 * 32 * 32, 2048),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(),

            BinarizeLinear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(),

            BinarizeLinear(2048, 2048),
            nn.Dropout(0.5),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(),

            nn.Linear(2048, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)  # convert 1*28*28 input tensor to 28*28 vector
        x = self.fc(x)
        return x
