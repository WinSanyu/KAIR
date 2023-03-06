import torch
from torch import nn
import utils.utils_image as util
from pnp.mixture_admm import *

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

class MixturePnP(nn.Module):

    def init_value(self, f):
        self.ind = ((f != 0) & (f != 255)).long()
        self.im_init = adpmedft(f) * (1 - self.ind) + f * self.ind
        x1 = adpmedft(f)
        x0 = adpmedft(f)
        y1 = f
        y0 = f
        z1 = f
        z0 = f
        gamma1 = torch.zeros_like(f)
        gamma0 = torch.zeros_like(f)
        S1 = torch.zeros_like(f)
        S0 = torch.zeros_like(f)
        W1 = torch.div(20, torch.abs(f - self.im_ini))
        W0 = torch.div(20, torch.abs(f - self.im_ini))
        return y1, x1, gamma1, S1, z1, W1, y0, x0, gamma0, S0, z0, W0

    def __init__(self, noise_level, beta, eta, admm_iter_num, eps, denoisor):
        super(MixturePnP, self).__init__()
        
        self.noise_level = noise_level
        self.beta = beta
        self.eta  = eta
        self.admm_iter_num = admm_iter_num
        # self.eps  = eps

        self.denoisor = denoisor

    # def ADMM(self, denoisor, y, sigma1, x1, gamma1, S1, z1):
    #     admm = lambda denoisor,y,a,b,c,d,e: (a,b,c,d,e)
    #     return admm(denoisor, y, sigma1, x1, gamma1, S1, z1)
    def ADMM(self, denoiser, ind, y, sigma, x, gamma, S, z, W, beta, eta):
        S = subproblem_S(y - x, 1./ W^2)
        z = subproblem_z(x, gamma, beta, eta)
        x = subproblem_x(beta, sigma, z, gamma, y, S)
        W = subproblem_W(y, x, S)
        sigma = subproblem_sigma(y, x, S, ind)
        gamma = subproblem_gamma(gamma, beta, x, z)
        beta = 1.2 * beta   # TODO finetune
        return y, sigma, x, gamma, S, z, W, beta, eta



    def forward(self, y, H=None):
        
        checkpoint = Intermediate(H)

        y1, x1, gamma1, S1, z1, W1, y0, x0, gamma0, S0, z0, W0 = self.init_value(y)
        sigma1, sigma0 = self.noise_level 
        beta1, beta0 = self.beta
        eta1, eta0 = self.eta
        for k in range(self.admm_iter_num):

            denoisor = self.denoisor.get_denoisor(k)
            if k >= 1:
                y1 = x1 + 0.5 * (y1 - x1) * self.ind - 0.2 * (self.im_init -  x1)*(1 - self.ind)
                y0 = y1
            
            y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2 = self.ADMM(
                self, denoisor, self.ind, y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1)

            if checkpoint.is_available():
                checkpoint.update(x2)

            y0, sigma0, x0, gamma0, S0, z0, W0, beta0, eta0 = y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1
            y1, sigma1, x1, gamma1, S1, z1, W1, beta1, eta1 = y2, sigma2, x2, gamma2, S2, z2, W2, beta2, eta2

        if checkpoint.is_available():
            x2 = checkpoint.get_best_measure_result()

        return x2 # / 255.