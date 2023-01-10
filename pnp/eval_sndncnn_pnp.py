import torch
import torch.nn as nn
import numpy as np
import sys
import os.path
from collections import OrderedDict
from torch.utils.data import DataLoader
sys.path.append("..") 

from utils import utils_image as util
from pnp.util_pnp import get_network_eval

class PNP_ADMM(nn.Module):
    def __init__(self, model, lamb, admm_iter_num, irl1_iter_num, mu, eps):
        super(PNP_ADMM, self).__init__()

        self.model = model

        self.lamb = lamb
        # self.denoisor_sigma = pnp_args['denoisor_sigma']
        self.admm_iter_num = admm_iter_num
        self.irl1_iter_num = irl1_iter_num
        self.mu = mu
        self.eps = eps

    # def model_forward(self, x):
    #     # TODO: denoisor_sigma
    #     predict = self.model(x)
    #     return predict

    def IRL1(self, f, u, v, b):
        for j in range(self.irl1_iter_num):
            # TODO: cal v
            pass
        v = u
        return v

    def ADMM(self, f, u, v, b):
        # model_input = f / 255.
        model_input = u / 255.
        u1 = self.model(model_input) * 255.
        b1 = b # self.mu
        v1 = self.IRL1(f, u1, v, b1)
        return u1, v1, b1

    def forward(self, f, GT=None):
        if GT is not None:
            best_result = None
            max_psnr = 0
            max_ssim = 0

        f *= 255
        u  = f
        v  = f
        b = torch.zeros(f.shape, device=f.device)

        for k in range(self.admm_iter_num):
            u1, v1, b1 = self.ADMM(f, u, v, b)

            if GT is not None: # only test
                cur_psnr, cur_ssim = self.get_intermediate_results(u1, GT)
                if max_psnr < cur_psnr:
                    max_psnr = cur_psnr
                    best_result = u1

                if max_ssim < cur_ssim:
                    max_ssim = cur_ssim

            u, v, b = u1, v1, b1

        if GT is not None:
            return best_result / 255.
        return u1 / 255.

    def get_intermediate_results(self, u1, img_H): # only test
        pre_i = torch.clamp(u1 / 255., 0., 1.)
        img_E = util.tensor2uint(pre_i)
        # img_H = util.tensor2uint(origin_img)
        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        return psnr, ssim

def eval(pnp_admm, H_paths, L_paths, noise_level, n_channels, device, logger):
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=1)
        img_L = util.uint2single(img_L)

        np.random.seed(seed=0)  # for reproducibility
        noise1 = np.random.normal(0, noise_level/255., img_L.shape)
        noise2 = np.random.normal(0, noise_level/255., img_L.shape)
        img_L = np.sqrt( (img_L + noise1)**2 + noise2**2 )

        # util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        # --------------------------------
        # (2) img_H
        # --------------------------------

        img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
        img_H = img_H.squeeze()

        # ------------------------------------
        # (3) img_E
        # ------------------------------------


        img_E = pnp_admm(img_L, img_H)

        img_E = util.tensor2uint(img_E)

        # --------------------------------
        # PSNR and SSIM
        # --------------------------------

        psnr = util.calculate_psnr(img_E, img_H, border=0)
        ssim = util.calculate_ssim(img_E, img_H, border=0)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    logger.info('Average PSNR/SSIM(RGB) - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(ave_psnr, ave_ssim))
    
    return ave_psnr, ave_ssim

def unpack_opt(opt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # test_loader = get_test_loader(opt)
    network = get_network_eval(opt)
    network.to(device)

    lamb = opt['pnp']['lamb']
    admm_iter_num = opt['pnp']['admm_iter_num']
    irl1_iter_num = opt['pnp']['irl1_iter_num']
    mu = opt['pnp']['mu']
    eps = opt['pnp']['eps']
    pnp_admm = PNP_ADMM(network, lamb, admm_iter_num, irl1_iter_num, mu, eps)

    H_paths = opt['datasets']['test']['dataroot_H']
    H_paths = util.get_image_paths(H_paths)
    L_paths = H_paths
    noise_level = opt['sigma_test']
    n_channels = opt['n_channels']
    return pnp_admm, H_paths, L_paths, noise_level, n_channels, device

if __name__ == "__main__":
    from pnp.util_pnp import get_opt, gen_logger
    opt = get_opt('../options/pnp/pnp_sndncnn.json')
    logger = gen_logger(opt)
    eval(*unpack_opt(opt), logger)
