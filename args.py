import argparse


parser = argparse.ArgumentParser(description='Basic parameters')

parser.add_argument('--device', type=str, default='cuda:0', help='The running device')
parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=24, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
parser.add_argument('--classes', type=int, default=1, help='Number of classes')
parser.add_argument('--img_size', type=int, default=224, help='Input patch size of network input')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--extended_pixels', type=int, default=10, help='The largest tumor extended pixels')
parser.add_argument('--record_save_path', type=str, default='./records/', help='Record saving path')
parser.add_argument('--model_save_path', type=str, default='./checkpoints/', help='Model saving path')
parser.add_argument('--visualization_save_path', type=str, default='./visualization/', help='Visualization result saving path')
parser.add_argument('--output_save_path', type=str, default='./outputs/', help='Output saving path')
parser.add_argument('--loss_weight', type=float, default=[0.5, 0.5], help='The loss weights')
parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')

args = parser.parse_args()
