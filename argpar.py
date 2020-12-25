import argparse

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--eval_time', default=2000, type=int, help='number of images between two evaluations')
parser.add_argument('--train_batch', default=2, type=int, help='train batchsize')
parser.add_argument('--val_batch', default=2, type=int, help='val batchsize')
parser.add_argument('--train_class_num', default=10, type=int, help='number of train class')
parser.add_argument('--gallery_size', default=10, type=int, help='size of gallery set')
parser.add_argument('--max_epoch', default=5, type=int, help='train epochs')
parser.add_argument('--is_simu', default=True, type=bool, help='train mode')
args, unknown = parser.parse_known_args()

def get_args():
    return args