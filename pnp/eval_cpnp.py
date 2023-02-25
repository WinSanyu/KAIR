import torch
import torch.nn as nn
import numpy as np
import sys
import os.path
from collections import OrderedDict
sys.path.append("..") 

from utils import utils_image as util
from pnp.util_pnp import get_network_eval, get_test_loader
from pnp.cpnp_admm import admm, Subproblem_mu

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

class PNP_ADMM(nn.Module):

    def init_fuvb(self, f):
        f *= 255
        u1 = f
        v1 = f
        b1 = torch.zeros(f.shape, device=f.device)
        u0 = None
        v0 = None
        b0 = None
        return f, u1, v1, b1, u0, v0, b0

    def __init__(self, model, sigma, lamb, admm_iter_num, irl1_iter_num, mu, rho, eps):
        super(PNP_ADMM, self).__init__()

        self.model = model

        self.sigma = sigma

        self.lamb = lamb
        self.admm_iter_num = admm_iter_num
        self.irl1_iter_num = irl1_iter_num
        self.mu0 = mu
        self.rho = rho
        self.eps = eps

    def ADMM(self, f, u1, v1, b1, u0, v0, b0, lamb, mu):
        return admm(self.model, 
                    f, u1, v1, b1, 
                    u0, v0, b0,
                    self.sigma, 
                    lamb, 
                    self.irl1_iter_num, 
                    mu,
                    self.eps)

    def forward(self, f, GT=None):
        ''' GT always is None.
        Only testing, GT is not None.
        '''
        checkpoint = Intermediate(GT)

        f, u1, v1, b1, u0, v0, b0 = self.init_fuvb(f)
        start_mu = 3
        mu = Subproblem_mu(self.mu0, self.rho, self.eps)

        for k in range(self.admm_iter_num):

            if k < start_mu:
                mu.disable()
            else:
                mu.enable()
            
            lamb = self.lamb
            u2, v2, b2 = self.ADMM(f, u1, v1, b1, u0, v0, b0, lamb, mu)

            # check intermediate results
            if checkpoint.is_available():
                checkpoint.update(u2)
            
            # err = ( torch.norm(u2 - u1,'fro').item() / 
            #         torch.norm(u1,'fro').item())
            # if err < 0.008:
            #     break

            u0, v0, b0 = u1, v1, b1
            u1, v1, b1 = u2, v2, b2

        if checkpoint.is_available():
            u2 = checkpoint.get_best_measure_result()

        return u2 / 255.

def eval(pnp_admm, test_loader, device, logger):
    idx = 0
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    for test_data in test_loader:
        idx += 1

        image_name_ext = os.path.basename(test_data['L_path'][0])
        
        img_L = test_data['L']
        img_H = test_data['H']

        img_L = img_L.to(device)
        img_E = pnp_admm(img_L, img_H)

        img_E = util.tensor2uint(img_E)
        img_H = util.tensor2uint(img_H)

        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)

        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB; SSIM: {:.4f}'.format(idx, image_name_ext, psnr, ssim))

        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))
    
    return ave_psnr, ave_ssim


def unpack_opt(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loader = get_test_loader(opt)
    network = get_network_eval(opt)
    network.to(device)

    sigma = opt['sigma']
    lamb = opt['pnp']['lamb']
    admm_iter_num = opt['pnp']['admm_iter_num']
    irl1_iter_num = opt['pnp']['irl1_iter_num']
    mu = opt['pnp']['mu']
    rho = opt['pnp']['rho']
    eps = opt['pnp']['eps']
    pnp_admm = PNP_ADMM(network, sigma, lamb, admm_iter_num, irl1_iter_num, mu, rho, eps)

    return pnp_admm, test_loader, device

if __name__ == "__main__":
    import argparse
    from pnp.util_pnp import get_opt, gen_logger
    json_path = '../options/pnp/pnp_sndncnn.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    json_path = parser.parse_args().opt

    opt = get_opt(json_path)
    logger = gen_logger(opt)
    eval(*unpack_opt(opt), logger)
