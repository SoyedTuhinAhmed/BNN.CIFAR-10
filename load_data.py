import torch
from torchvision import datasets, transforms


class LoadData():
    def __init__(self, kwargs, batch_size, test_batch_size):
        self.kwargs = kwargs
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.train_loader = 0
        self.test_loader = 0

    def mnist(self):
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=self.test_batch_size, shuffle=True, **self.kwargs)

    def cifar10(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                  shuffle=True, **self.kwargs)

        testset = datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                 shuffle=False, **self.kwargs)
