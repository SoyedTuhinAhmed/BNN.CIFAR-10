import torch
import torch.nn as nn
import torch.optim as optim
from argument_parser import *
from binarylenet5 import *
from lenet5 import *
from alexnet import *
from train import *
from test import *
from load_data import *
import os


def init():
    pass


def main():
    arg_parser = ArgumentParser()
    args = arg_parser.parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    path_nn = "results/" + str(args.seed) + "_" + str(args.alpha)
    if args.humult != 8.0:
        path_nn += "_" + str(args.humult)

    path_nn += "/"

    path_ev = path_nn + str(args.mix) + "_" + str(args.refresh) + "_" + str(args.d1) + "_" + str(args.d2)

    if args.dynrefresh > 0.0:
        path_ev += "_dyn_" + str(args.dynrefresh) + "_" + args.colrefresh

    if args.colrefresh == "th":
        path_ev += "_" + str(args.dyncolthr)
    if args.colrefresh == "tri":
        path_ev += "_tri_" + str(args.refresh)

    path_ev += "/"

    log_path = path_ev + "log.txt"

    print(path_ev)

    if not os.path.exists(path_nn):
        os.mkdir(path_nn)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    # ------------------------------------------------------------------------------------------------------------------
    """
                                                initialize
    """
    lenet5 = BinaryLeNet5(args.humult)
    # lenet5 = LeNet5(args.humult)
    # lenet5 = AlexNet()

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        lenet5.to(device)  # if CUDA available then run on GPU otherwise on CPU

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    optimizer = optim.Adam(lenet5.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    data = LoadData(kwargs, batch_size=args.batch_size, test_batch_size=args.test_batch_size)
    data.cifar10()
    nn_train = Train(model=lenet5, optimizer=optimizer, train_loader=data.train_loader, criterion=criterion,
                     cuda_arg=args.cuda, log_interval_arg=args.log_interval)
    test = Test(model=lenet5, cuda_args=args.cuda, criterion=criterion, test_loader=data.test_loader)

    """
                                                run training
    """
    best_accuracy = 0
    best_ones = list(lenet5.parameters())[0].size(0) * list(lenet5.parameters())[0].size(1)
    for epoch in range(1, args.epochs + 1):
        print('Epoch: ' + str(epoch) + "  ", end='')

        nn_train.another_train(epoch)
        test_accuracy, test_ones = test.nn_test()

        if epoch % 10 == 0:
           optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1

        best_ones = list(lenet5.parameters())[0].size(0) * list(lenet5.parameters())[0].size(1)

        if best_ones > test_ones:  # if number of ones decrease in current ireration/run then save current no of ones
            best_ones = test_ones
            torch.save(lenet5.state_dict(), path_nn + "ones_run_customLoss.pt")
            print("----------> Updated best ones run")

        if best_accuracy < test_accuracy:  # if accuracy decreases in current ireration/run then save best accuracy
            best_accuracy = test_accuracy
            torch.save(lenet5.state_dict(), path_nn + "accuracy_run_customLoss.pt")
            print("----------> Updated best accuracy run")


if __name__ == '__main__':
    main()
