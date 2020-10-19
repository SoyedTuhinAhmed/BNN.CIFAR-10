import torch.nn as nn
from binarized_modules import BinarizeLinear, BinarizeConv2d


class BinAlexNet(nn.Module):
    def __init__(self):
        super(BinAlexNet, self).__init__()
        self.features = nn.Sequential(
            BinarizeConv2d(3, 64, kernel_size=5, padding=1, bias=False),  # out dim = 64*30*30
            nn.MaxPool2d(kernel_size=2, stride=2),  # out dim = 64 * 15* 15
            nn.BatchNorm2d(64),
            nn.Hardtanh(),

            BinarizeConv2d(64, 192, kernel_size=5, padding=1, bias=False),  # out dim = 192 * 13 * 13
            nn.MaxPool2d(kernel_size=2, stride=2),  # out dim = 192 * 6 * 6
            nn.BatchNorm2d(192),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(192, 384, kernel_size=3, padding=1, bias=False),  # out dim = 384 * 6 * 6
            nn.BatchNorm2d(384),
            nn.Hardtanh(),

            BinarizeConv2d(384, 256, kernel_size=3, padding=1, bias=False),  # out dim = 256 * 6 * 6
            nn.BatchNorm2d(256),
            nn.Hardtanh(),

            BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=False),  # out dim = 256 * 6 * 6
            nn.MaxPool2d(kernel_size=3, stride=2),  # out dim = 256 * 4 * 4
            nn.BatchNorm2d(256),
            nn.Hardtanh()
        )

        self.classifier = nn.Sequential(
            BinarizeLinear(256 * 2 * 2, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.Hardtanh(),
            # nn.Dropout(0.5),

            BinarizeLinear(4096, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.Hardtanh(),
            # nn.Dropout(0.5),

            BinarizeLinear(2048, 10)
        )

    def forward(self, x):
        x = self.features(x)
        # x = x.flatten()
        x = x.view(-1, 256 * 2 * 2)
        x = self.classifier(x)
        return x
