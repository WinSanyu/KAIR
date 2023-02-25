import torch
from torch import nn
from pnp.cpnp_admm import admm, Subproblem_mu
from utils import utils_image as util

class Intermediate:
    def __init__(self, GT):
        self.available = not GT is None
        if self.available:
            self.GT = GT
            self.best_psnr_result = None
            self.best_ssim_result = None
            self.best_measure_result = None
            self.max_psnr = 0
            self.max_ssim = 0
            self.max_measure = 0
    
    def _measure_psnr_ssim(self, psnr, ssim):
        measure = psnr + 10*ssim
        return measure

    def is_available(self):
        return self.available

    def update(self, u):
        if not self.is_available():
            return None

        cur_psnr, cur_ssim = self._get_intermediate_results(u, self.GT)

        if self.max_psnr < cur_psnr:
            self.max_psnr = cur_psnr
            self.best_psnr_result = u   

        if self.max_ssim < cur_ssim:
            self.max_ssim = cur_ssim
            self.best_ssim_result = u

        cur_measure = self._measure_psnr_ssim(cur_psnr, cur_psnr)
        if self.max_measure < cur_measure:
            self.max_measure = cur_measure
            self.best_measure_result = u
    
    def get_best_psnr_result(self):
        return self.best_psnr_result
    
    def get_best_ssim_result(self):
        return self.best_ssim_result
    
    def get_best_measure_result(self):
        return self.best_measure_result
   
    def _get_intermediate_results(self, u1, img_H):
        pre_i = torch.clamp(u1 / 255., 0., 1.)
        img_E = util.tensor2uint(pre_i)
        img_H = util.tensor2uint(img_H)
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        return psnr, ssim

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