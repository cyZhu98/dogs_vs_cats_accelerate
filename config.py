import argparse

parser = argparse.ArgumentParser(description='Dogs VS Cats Classification')
parser.add_argument('--DIR', default='..', help='dataset location')
parser.add_argument('--split_ratio', type=float, default=0.8,
                    help='default setting. train : val = 8 : 2')

parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('-b', '--batch-size', default=32,
                    type=int, help='mini-batch size, 8.7G memory per gpu with bs32, swinB 224')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')

parser.add_argument('-p', '--print_freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('-s', '--save', action='store_true',
                    help='save best model')


def return_args():
    args = parser.parse_args()
    return args
