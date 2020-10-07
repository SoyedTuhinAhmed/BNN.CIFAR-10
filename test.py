from torch.autograd import Variable
import torch
from models.binarized_modules import *


class Test():
    def __init__(self, model, test_loader, cuda_args, criterion):
        self.test_loader = test_loader
        self.cuda_args = cuda_args
        self.model = model
        self.criterion = criterion

    def ones(self):
        params = list(self.model.parameters())
        t = Binarize(params[0])
        t += 1
        t = torch.reshape(t, (-1,))
        t = torch.sum(t)
        return t.item() / 2

    def nn_test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                if self.cuda_args:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = self.model(data)

                params = list(self.model.parameters())
                weights = params[0]

                test_loss += self.criterion(output, target, weights).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        test_accuracy = 100. * correct / len(self.test_loader.dataset)
        test_ones = Test.ones()
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Ones: {:.1f}'.format(
            test_loss, correct, len(self.test_loader.dataset),
            test_accuracy,
            test_ones))
        return test_accuracy, test_ones
