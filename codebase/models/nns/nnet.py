import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F

class GlobalEncoder(nn.Module):
    def __init__(self, x_dim, z_dim, z_num):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.z_num = z_num
        self.net = nn.Sequential(
            nn.Linear(x_dim, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Linear(512, z_dim*z_num)
        )

    def encode(self, x):
        return self.net(x)

class LocalEncoder(nn.Module):
    def __init__(self, z_dim, z_num):
        super().__init__()
        self.z_dim = z_dim
        self.z_num = z_num
        self.net = nn.Sequential(
            nn.Linear(z_dim*z_num, z_dim*z_num),
            nn.BatchNorm1d(z_dim*z_num),
            nn.ReLU(),
            nn.Linear(z_dim*z_num, 2*z_dim)
        )

    def encode(self, input):
        h = self.net(input)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, input_dim, x_dim):
        super().__init__()
        self.input_dim = input_dim
        self.x_dim = x_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, x_dim)
        )

    def decode(self, z):
        return self.net(z)

class Mu(nn.Module):
    def __init__(self, mu, epoch=0):
        super().__init__()
        self.mu = mu
        self.epoch = epoch

    def sample(self, hard=True):
        tau = .99**self.epoch
        logits = torch.autograd.Variable(torch.stack((self.mu,1-self.mu),dim=1), requires_grad=True)
        c = F.gumbel_softmax(logits, tau=tau, hard=hard)[:,0]
        c.retain_grad()
        return c