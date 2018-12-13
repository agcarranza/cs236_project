import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class GVAE(nn.Module):
    def __init__(self, x_dim, z_dim, z_num, nn='nnet', name='gvae'):
        super().__init__()
        self.name = name
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.z_num = z_num
        nn = getattr(nns, nn)
        self.dec = nn.Decoder(z_dim*z_num,x_dim)
        self.gl_enc = nn.GlobalEncoder(x_dim, z_dim, z_num)
        self.bu_enc = []
        self.td_enc = []
        for n in range(z_num):
            self.bu_enc.append(nn.LocalEncoder(z_dim,z_num))
            self.td_enc.append(nn.LocalEncoder(z_dim,z_num))
        self.mu = nn.Mu(torch.zeros(z_num*(z_num-1)//2))
        # self.mu = torch.nn.Parameter(torch.zeros(z_num*(z_num-1)//2))

    def nelbo(self, x, epoch=None):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        # get dimensions
        z_dim = self.z_dim
        z_num = self.z_num
        b_size = x.size(0)

        # sample c and determine parents
        c = self.mu.sample()
        mask = torch.zeros((z_num,z_num))
        mask[torch.tril(torch.ones((z_num,z_num)),-1)==1] = c
        p_num = (mask!=0).sum(dim=0)

        # get data encoding
        hx = self.gl_enc.encode(x)

        # compute log prior and log posterior
        # sample z from posterior
        logq = torch.zeros(b_size)
        logp = torch.zeros(b_size)
        z = torch.zeros((b_size, z_dim, z_num))
        for n in range(1, z_num+1):
            # if no parents, sample unit gaussian
            # else, sample encoded gaussian
            if p_num[-n]==0:
                m_n = torch.zeros((b_size,z_dim), requires_grad=False)
                v_n = torch.ones((b_size,z_dim), requires_grad=False)
                z[:,:,-n] = ut.sample_gaussian(m_n, v_n)
            else:
                # get parents of z_n
                p_n = z.transpose(1,2).reshape(-1,z_dim*z_num)

                # get bottom-up and top-down encoded gaussian parameters
                bu_psi_n = self.bu_enc[-n].encode(hx)  
                td_psi_n = self.td_enc[-n].encode(p_n)

                # compute precision-weighted fusion of encoded gaussian parameters
                psi_n = self.gaussian_params_fusion(bu_psi_n,td_psi_n)

                # sample z_n from posterior
                z_n = ut.sample_gaussian(psi_n[0],psi_n[1])
                z[:,:,-n] = z_n

                # add to log prior and log posterior
                logp += ut.log_normal(z_n, td_psi_n[0], td_psi_n[1])
                logq += ut.log_normal(z_n, psi_n[0], psi_n[1])

        # compute log conditional
        logits = self.dec.decode(hx)
        # logits = self.dec.decode(z.transpose(1,2).reshape(-1,z_dim*z_num))
        logp_cond = ut.log_bernoulli_with_logits(x, logits)

        # compute rec and kl terms
        rec = -logp_cond.mean()
        kl = (logq-logp).mean()
        nelbo = rec + kl
        # print(nelbo.data, kl.data, rec.data)
        # print(c)
        return nelbo, kl, rec

    def loss(self, x, epoch=None):
        nelbo, kl, rec = self.nelbo(x, epoch)
        mu = self.mu.mu.detach().numpy()
        loss = nelbo
        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
            ('train/mu', mu)
        ))
        return loss, summaries

    def gaussian_params_fusion(self, bu_psi, td_psi):
        m_bu, v_bu = bu_psi
        m_td, v_td = td_psi
        v = 1 / (v_bu+v_td)
        m = (m_bu*v_bu + m_td*v_td) * v
        return m, v

    # def sample_c(self, epoch=None):
    #     tau = .99**epoch if epoch else 1.0
    #     l1 = F.logsigmoid(self.mu)
    #     l2 = 1.0-l1
    #     logits = torch.stack((l1,l2),dim=1)
    #     return F.gumbel_softmax(logits, tau=tau, hard=True)[:,0]

    # def sample_c(self, epoch=None):
    #     tau = .99**epoch if epoch else 1.0
    #     logits = torch.autograd.Variable(torch.stack((self.mu,1-self.mu),dim=1),requires_grad=True)
    #     c = F.gumbel_softmax(logits, tau=tau, hard=True)[:,0]
    #     return c

    def sample_z(self, b_size):
        z_dim = self.z_dim
        z_num = self.z_num
        with torch.no_grad():
            # sample c and get parents
            c = self.sample_c()
            mask = torch.zeros((z_num,z_num))
            mask[torch.tril(torch.ones((z_num,z_num)),-1)==1] = c
            p_num = (mask!=0).sum(dim=0)
            z = torch.zeros((b_size,z_dim,z_num))
            for n in range(1, z_num+1):
                # if no parents, sample unit gaussian
                # else, sample encoded gaussian
                if p_num[-n]==0:
                    z[:,:,-n] = ut.sample_gaussian(torch.zeros((b_size,z_dim)), torch.ones((b_size,z_dim)))
                else:
                    p_n = z.transpose(1,2).reshape(-1,z_dim*z_num)          # get parents of z_n
                    td_psi_n = self.td_enc[-n].encode(p_n)                  # get top-down encoded gaussian parameters
                    z[:,:,-n] = ut.sample_gaussian(td_psi_n[0],td_psi_n[1]) # sample z_n

        return z.transpose(1,2).reshape(-1,z_dim*z_num)

    def sample_x(self, b_size):
        z = self.sample_z(b_size)
        logits = self.dec.decode(z)
        return torch.bernoulli(torch.sigmoid(logits))

    # def sample_sigmoid(self, batch):
    #     z = self.sample_z(batch)
    #     return self.compute_sigmoid_given(z)

    # def compute_sigmoid_given(self, z):
    #     logits = self.dec.decode(z)
    #     return torch.sigmoid(logits)

    # def sample_z(self, batch):
    #     return ut.sample_gaussian(
    #         self.z_prior[0].expand(batch, self.z_dim),
    #         self.z_prior[1].expand(batch, self.z_dim))

    # def sample_x(self, batch):
    #     z = self.sample_z(batch)
    #     return self.sample_x_given(z)

    # def sample_x_given(self, z):
    #     return torch.bernoulli(self.compute_sigmoid_given(z))
