import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')
parser.add_argument('--crop_size', type=int, default=64, help='crop size for image')
parser.add_argument('--num_epochs', type=int, default=100, help='total epoch for training')
parser.add_argument('--print_every', type=int, default=100, help='print statistics for every default iteration')
parser.add_argument('--save_every', type=int, default=10, help='save model weights for every default epoch')
parser.add_argument('--noise_dim', type=int, default=100, help='noise dimension')
parser.add_argument('--limit', type=int, default=1e5, help='the number of images to be generated for inference')

parser.add_argument('--D_lr', type=float, default=4e-4, help='learning rate for Discriminator')
parser.add_argument('--G_lr', type=float, default=1e-4, help='learning rate for Generator')
parser.add_argument('--lr_decay_rate', type=float, default=0.5, help='decay learning rate')
parser.add_argument('--lr_decay_every', type=int, default=50, help='decay learning rate for every default epoch')
parser.add_argument('--lr_scheduler', type=str, default='step', help='learning rate scheduler, options: [Step, Plateau, Cosine]')

parser.add_argument('--celeba_path', type=str, default='./celeba/', help='CelebA data path')
parser.add_argument('--samples_path', type=str, default='./results/samples/', help='samples path')
parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
parser.add_argument('--plots_path', type=str,  default='./results/plots/', help='plots path')
parser.add_argument('--inference_path', type=str,  default='./results/inference/', help='inference path')

parser.add_argument('-f')
config = parser.parse_args()

if __name__ == "__main__":
    print(config)