from torch.autograd import Variable


class Train():
    def __init__(self, model, optimizer, criterion, cuda_arg, log_interval_arg, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.cuda_arg = cuda_arg
        self.criterion = criterion
        self.optimizer = optimizer
        self.log_interval_arg = log_interval_arg

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda_arg:
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

            if self.log_interval_arg > 0 and batch_idx % self.log_interval_arg == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.data.item()))

            # list(model.parameters())[0] -= 0.001
