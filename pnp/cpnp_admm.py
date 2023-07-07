import torch

class Subproblem_mu:
    '''All operations related to mu'''
    def __init__(self, mu, rho, eps):
        self.available = False
        self.rho = rho
        self.mu = mu
        self.eps = eps

    def disable(self):
        self.available = False

    def enable(self):
        self.available = True

    def get_mu(self):
        if not self.available:
            return None
        return self.mu

    def updata_mu(self, x):
        if not self.available:
            return None
        rhomu1 = self.rho * self.mu
        self.mu = rhomu1 * x
        return self.mu
    
    def is_available(self):
        return self.available

def SpecialB(x):
    return torch.special.i1e(x) / torch.special.i0e(x)

def _irl1(f, u, v, b, sigma, lamb, irl1_iter_num):
    irl1_input = u + b
    beta = 1
    f_sigma2 = f / sigma**2
    for _ in range(irl1_iter_num):
        Iz = SpecialB(f_sigma2 * v)
        Iz = torch.clamp(Iz, min=0) # min=0
        y = f_sigma2 * (1 - Iz)
        v = (lamb*f_sigma2 + beta*irl1_input - lamb*y) / (lamb/sigma**2 + beta)
    return v

def _fast_irl1(f, u, v, b, sigma, lamb, irl1_iter_num):
    irl1_input = u + b
    f_sigma2 = f / sigma**2
    lamb_f_sigma2 = lamb * f_sigma2
    lamb_sigma2_beta = lamb/sigma**2 + 1
    for _ in range(irl1_iter_num):
        Iz = SpecialB(f_sigma2 * v)
        Iz = torch.clamp(Iz, min=0)
        y = f_sigma2 * (1 - Iz)
        v = (lamb_f_sigma2 + irl1_input - lamb*y) / lamb_sigma2_beta
    return v

def _bregman(mu, b1, u2, u1, v1, v0, eps):
    if not mu.is_available():
        b2 = b1 + u1 - v1
        return b2

    tmp = (torch.norm(u1 - v0,'fro').item() 
        / max(eps, torch.norm(u2 - v1,'fro').item()) )
    mu2 = mu.updata_mu(tmp)

    b2 = b1 + u1 - u2 + 2*mu2*(u2 - v1)
    return b2

def admm(model, f, u1, v1, b1, u0, v0, b0, sigma, lamb, irl1_iter_num, mu, eps):
    '''CPnP'''
    model_input = (2*v1 - b1 - u1) / 255.
    u2 = model(model_input) * 255.
    u2 = torch.clamp(u2, min=0., max=255.)
    b2 = _bregman(mu, b1, u2, u1, v1, v0, eps)
    v2 = _fast_irl1(f, u2, v1, b2, sigma, lamb, irl1_iter_num)
    return u2, v2, b2

def admm1(model, f, u1, v1, b1, sigma, lamb, irl1_iter_num, eps):
    '''CPnP2'''
    model_input = (2*v1 - b1 - u1) / 255.
    u2 = model(model_input) * 255.
    u2 = torch.clamp(u2, min=0., max=255.)
    b2 = b1 + u1 - v1
    v2 = _irl1(f, u2, v1, b2, sigma, lamb, irl1_iter_num)
    return u2, v2, b2

