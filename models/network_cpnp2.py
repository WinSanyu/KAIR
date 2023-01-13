import torch
from torch import nn
from models.network_dncnn import DnCNN as Net
from pnp.cpnp_admm import admm

class CPnP2(nn.Module):
    def __init__(self, sigma, lamb, admm_iter_num, irl1_iter_num, mu, eps):
        super(CPnP2, self).__init__()

        self.sigma = sigma

        self.lamb = lamb
        self.admm_iter_num = admm_iter_num
        self.irl1_iter_num = irl1_iter_num
        self.mu = mu
        self.eps = eps

        for i in range(admm_iter_num):
            setattr(self, 'model' + str(i), 
                Net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='BR')
            )

    def ADMM(self, model, f, u, v, b):
        return admm(model, f, u, v, b, 
                    self.sigma, self.lamb, 
                    self.irl1_iter_num, self.eps)

    def forward(self, f):
        f *= 255
        u  = f
        v  = f
        b = torch.zeros(f.shape, device=f.device)

        for k in range(self.admm_iter_num):
            model = getattr(self, 'model' + str(k))
            u1, v1, b1 = self.ADMM(model, f, u, v, b)
            u, v, b = u1, v1, b1

        return u1 / 255.