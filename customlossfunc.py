import torch.nn as nn


class CustLoss(nn.CrossEntropyLoss):
    """
    apply modified CrossEntropyLoss loss function to reduce number of ones as proposed in the papaer.
    """

    def __init__(self, args_alpha, modified=True):
        super(CustLoss, self).__init__()
        self.modified = modified
        self.args_alpha = args_alpha

    def forward(self, output, target, weights):
        """
        weights.clamp(-1, 1) makes anything <-1 --> -1 and anything > 1 --> 1 so all weights are betwn (-1,1) and add
        should make all weights >= 0, sum add up all the W and div to get average of individual W

        :param output:
        :param target:
        :param weights:
        :return:
        """
        return nn.CrossEntropyLoss.forward(self, output, target) + self.args_alpha * weights.clamp(-1, 1).add(1).sum().div(
            2 * weights.size(0) * weights.size(1))
