import torch
from torch import nn
from pnp.cpnp_admm import admm, Subproblem_mu

class CPnP2(nn.Module):

    def init_fuvb(self, f):
        f *= 255
        u1 = f
        v1 = f
        b1 = torch.zeros(f.shape, device=f.device)
        u0 = None
        v0 = None
        b0 = None
        return f, u1, v1, b1, u0, v0, b0

    def __init__(self, sigma, lamb, admm_iter_num, irl1_iter_num, mu, eps, rho, denoisor):
        super(CPnP2, self).__init__()

        self.sigma = sigma
        self.lamb = torch.nn.Parameter(lamb*torch.ones(admm_iter_num), requires_grad=True)
        self.admm_iter_num = admm_iter_num
        self.irl1_iter_num = irl1_iter_num
        self.mu0 = mu
        self.rho = rho
        self.eps = eps
        self.denoisor = denoisor

    def ADMM(self, model, f, u1, v1, b1, u0, v0, b0, lamb, mu):
        return admm(model, f, 
                    u1, v1, b1, 
                    u0, v0, b0,
                    self.sigma, lamb, 
                    self.irl1_iter_num,
                    mu, 
                    self.eps)

    def forward(self, f):
        f, u1, v1, b1, u0, v0, b0 = self.init_fuvb(f)
        start_mu = 3
        mu = Subproblem_mu(self.mu0, self.rho, self.eps)

        for k in range(self.admm_iter_num):

            if k < start_mu:
                mu.disable()
            else:
                mu.enable()

            lamb = self.lamb[k]
            model = self.denoisor.get_denoisor(k)
            
            u2, v2, b2 = self.ADMM(model, f, u1, v1, b1, u0, v0, b0, lamb, mu)

            u0, v0, b0 = u1, v1, b1
            u1, v1, b1 = u2, v2, b2

        return u2 / 255.