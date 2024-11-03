import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='...', help='The directory of the pretrained dataset')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=2e-3, help='segmentation network learning rate')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
parser.add_argument('--num_folds', type=int, default=5, help='The number of fold')

args = parser.parse_args()