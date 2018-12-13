import argparse
import numpy as np
import os
import tensorflow as tf
import torch
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

def train(model, train_loader, labeled_subset, device, tqdm, writer,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    i = 0
    epoch_count = 0
    mus = []
    its = []
    with tqdm(total=iter_max) as pbar:
        while True:
            for batch_idx, (xu, yu) in enumerate(train_loader):
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()

                xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                yu = yu.new(np.eye(10)[yu]).to(device).float()
                loss, summaries = model.loss(xu, epoch_count)

                loss.backward()
                # for name, param in model.named_parameters():
                #     if name in ['mu']:
                #         param.retain_grad()
                #         print(param.requires_grad, param.grad)

                # # start debugger
                # import pdb; pdb.set_trace()            
                optimizer.step()

                # Feel free to modify the progress bar
                pbar.set_postfix(loss='{:.2e}'.format(loss))
                pbar.update(1)

                # Log summaries
                if i % 50 == 0:
                    ut.log_summaries(writer, summaries, i)
                    for name, param in model.named_parameters():
                        if name in ['mu']:
                            mus.append(param)
                            its.append(i)
                    
                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i)

                if i == iter_max:
                    plot_mus(its,mus)
                    return
            epoch_count += 1

def plot_mus(its,mus):
    its = np.array(its)
    mus = torch.stack(mus)
    z_num = int((1+np.sqrt(1+8*mus.size(1)))/2)
    ind = (torch.tril(torch.ones((z_num,z_num)),-1)==1).nonzero()+1
    for n in range(mus.size(1)):
        i, j = ind[n].detach().numpy()
        symb = str(i)+','+str(j)
        data = mus[:,n].detach().numpy()
        plt.plot(its, data, label=r'$\mu_{{}}$'.format(symb), linewidth=2)
    plt.legend(ncol=3)
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.savefig('./img/mu.png')
    return
