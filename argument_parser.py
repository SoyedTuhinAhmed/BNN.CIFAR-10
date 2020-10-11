import argparse


class ArgumentParser:
    def __init__(self):
        # Training settings
        self.parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        self.parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                            help='input batch size for training (default: 256)')
        self.parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                            help='input batch size for testing (default: 1000)')
        self.parser.add_argument('--epochs', type=int, default=100, metavar='N',
                            help='number of epochs to train (default: 10)')
        self.parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                            help='learning rate (default: 0.001)')
        self.parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        self.parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        self.parser.add_argument('--gpus', default=3,
                            help='gpus used for training - e.g 0,1,3')
        self.parser.add_argument('--log-interval', type=int, default=0, metavar='N',
                            help='how many batches to wait before logging training status')
        self.parser.add_argument('--eval', action='store_true', default=False,
                            help='Only evaluate the previous training (default: 0)')
        self.parser.add_argument('--eval-ones', action='store_true', default=False,
                            help='Evaluate the best ones-run or the best accuracy(default)')
        self.parser.add_argument('--useHighRet', action='store_true', default=False,
                            help='Use high retention mitigation')

        self.parser.add_argument('--onlyRefresh', action='store_true', default=False,
                            help='Only output dynamic refresh calculation results')

        self.parser.add_argument('--d1', type=float, default=60.0, metavar='M',
                            help='delta1 (default: 60)')
        self.parser.add_argument('--d2', type=float, default=60.0, metavar='M',
                            help='delta2 (default: 60)')
        self.parser.add_argument('--mix', type=float, default=0.0, metavar='M',
                            help='Mix of delta1 and delta2 [0-100] (default: 0, all delta1)')
        self.parser.add_argument('--refresh', type=float, default=0.0, metavar='M',
                            help='Percentage of rows to refresh in the high retention subarray [0-100] (default: 0, no refresh)')
        self.parser.add_argument('--dynrefresh', type=float, default=0.0, metavar='M',
                            help='Threshold for calculating the dynamic row refresh [0.0-1.0] (default: 0.0, no dynamic refresh)')
        self.parser.add_argument('--alpha', type=float, default=0.0, metavar='N',
                            help='weight of regulation (default: 0 - no regulation)')
        self.parser.add_argument('--colrefresh', type=str, default="all",
                            help='Column refresh strategy (default: all)')
        self.parser.add_argument('--dyncolthr', type=float, default=0.0, metavar='M',
                            help='Threshold for calculating the dynamic col refresh [0.0-1.0] if colrefresh strategy is "th" (default: 0.0)')
        self.parser.add_argument('--humult', type=float, default=1.0, metavar='M',
                            help='Hidden unit multiplier (default: 8.0)')
