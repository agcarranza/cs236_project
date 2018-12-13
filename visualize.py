import argparse
import numpy as np
import torch
from codebase import utils as ut
from codebase.models.gvae import GVAE
from pprint import pprint
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

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
gvae = GVAE(x_dim=args.x_dim, z_dim=args.z_dim, z_num=args.z_num, name=model_name).to(device)
ut.load_model_by_name(gvae, global_step=args.iter_max)

num_img = 200
X = gvae.sample_x(num_img)
X = X.reshape(200,1,28,28)
utils.save_image(X, "./img/mnist.png", nrow=20)
img = utils.make_grid(X, nrow=20)
plt.imshow(img.detach().numpy().transpose((1, 2, 0)))
plt.show()



# Bonus
# step = 10000 * 47
# num_z = 20
# digit = 0
# layout = [
#     ('model={:s}',  'fsvae'),
#     ('run={:04d}', args.run)
# ]
# model_name = '_'.join([t.format(v) for (t, v) in layout])
# fsvae = FSVAE(name=model_name).to(device)
# ut.load_model_by_name(fsvae, global_step=step)

# # z = fsvae.sample_z(1)
# # y = torch.zeros_like(z)
# # y[0,digit] = 1.
# # theta_m = fsvae.dec.decode(z, y)
# # img = torch.clamp(theta_m, 0, 1)
# # img = img.reshape(32,32,3)
# # plt.imshow(img.detach().numpy())
# # plt.show()

# z = fsvae.sample_z(num_z) 
# y = np.repeat(np.arange(10), z.size(0))
# y = z.new(np.eye(10)[y])
# z = ut.duplicate(z, 10)
# theta_m = fsvae.dec.decode(z, y).clamp(0, 1)
# x = theta_m.reshape(200,3,32,32)
# utils.save_image(x, "bonus.png", nrow=20)
# img = utils.make_grid(x, nrow=20)
# plt.imshow(img.detach().numpy().transpose((1,2,0)))
# plt.show()


