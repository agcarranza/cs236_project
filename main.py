import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.gvae import GVAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--x_dim',     type=int, default=28*28, help="Number of observable dimensions")
parser.add_argument('--z_dim',     type=int, default=32,    help="Number of latent dimensions per node")
parser.add_argument('--z_num',     type=int, default=10,    help="Number of latent nodes")
parser.add_argument('--iter_max',  type=int, default=20000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
args = parser.parse_args()
layout = [
    ('model={:s}',   'gvae'),
    ('x-dim={:03d}', args.x_dim),
    ('z-dim={:02d}', args.z_dim),
    ('z-num={:02d}', args.z_num),
    ('run={:04d}',   args.run)
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
gvae = GVAE(x_dim=args.x_dim, z_dim=args.z_dim, z_num=args.z_num, name=model_name).to(device)

if args.train:
    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    train(model=gvae,
          train_loader=train_loader,
          labeled_subset=labeled_subset,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)
    ut.evaluate_lower_bound(gvae, labeled_subset)
else:
    ut.load_model_by_name(gvae, global_step=args.iter_max)
    ut.evaluate_lower_bound(gvae, labeled_subset)