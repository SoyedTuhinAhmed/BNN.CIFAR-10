from covnet import *
from argument_parser import *
import torch.nn as nn
import torch.optim
from load_data import *
from torch.autograd import Variable


class Train():
    def __init__(self):
        self.model = CovNet()
        self.data_loader = LoadData()
        self.argparse = ArgumentParser()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.argparse.args.lr)

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            if self.argparse.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)

            params = list(self.model.parameters())
            weights = (params[0])

            loss = self.criterion(output, target, weights)

            # if epoch%40==0:
            #    optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

            self.optimizer.zero_grad()
            loss.backward()
            for p in list(self.model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
            self.optimizer.step()
            for p in list(self.model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data.clamp_(-1, 1))

            if self.argparse.args.log_interval > 0 and batch_idx % self.argparse.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.data.item()))

            # list(model.parameters())[0] -= 0.001
