import torch
from torch import nn
import utils.utils_image as util
# from pnp.mixture_admm import admm

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
        z1 = f
        z0 = f
        x1 = f
        x0 = f
        gamma1 = torch.zeros_like(f)
        gamma0 = torch.zeros_like(f)
        sigma1 = torch.zeros_like(f)
        sigma0 = torch.zeros_like(f)
        S1 = torch.zeros_like(f)
        S0 = torch.zeros_like(f)
        return sigma1, x1, gamma1, S1, z1, sigma0, x0, gamma0, S0, z0

    def __init__(self, beta, eta, admm_iter_num, eps, denoisor):
        super(MixturePnP, self).__init__()

        self.beta = beta
        self.eta  = eta
        self.admm_iter_num = admm_iter_num
        self.eps  = eps

        self.denoisor = denoisor

    def ADMM(self, denoisor, y, sigma1, x1, gamma1, S1, z1):
        admm = lambda denoisor,y,a,b,c,d,e: (a,b,c,d,e)
        return admm(denoisor, y, sigma1, x1, gamma1, S1, z1)

    def forward(self, y, H=None):
        
        checkpoint = Intermediate(H)

        sigma1, x1, gamma1, S1, z1, sigma0, x0, gamma0, S0, z0 = self.init_value(y)

        for k in range(self.admm_iter_num):

            denoisor = self.denoisor.get_denoisor(k)
            
            sigma2, x2, gamma2, S2, z2 = self.ADMM(
                denoisor, y, sigma1, x1, gamma1, S1, z1)

            if checkpoint.is_available():
                checkpoint.update(x2)

            sigma0, x0, gamma0, S0, z0 = sigma1, x1, gamma1, S1, z1
            sigma1, x1, gamma1, S1, z1 = sigma2, x2, gamma2, S2, z2

        if checkpoint.is_available():
            x2 = checkpoint.get_best_measure_result()

        return x2 # / 255.