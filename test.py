from load_data import *
import torch


class Test():
    def __init__(self):
        self.data = LoadData()

    def nn_test(self):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data.test_loader.dataset),
            100. * correct / len(self.data.test_loader.dataset)))
