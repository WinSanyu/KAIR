# import matplotlib.pyplot as plt
# from torchvision.transforms import ToPILImage
# from tqdm import tqdm
import logging
from copy import deepcopy
import os.path
import sys
sys.path.append("..") 

from utils import utils_option as option
from utils import utils_image as util
from utils import utils_logger
from pnp.eval_sndncnn_pnp import unpack_opt, eval

def get_opt(json_path):
    opt = option.parse(json_path, is_train=False)
    opt = option.dict_to_nonedict(opt)
    return opt

def gen_opts(json_path):
    '''Generate all opt grids to be searched'''
    opts = []
    all_opt = get_opt(json_path)
    pnp_opt = all_opt['pnp']
    for lamb in range(*pnp_opt['lamb']):
        for denoisor_sigma in pnp_opt['denoisor_sigma']:
            opt = deepcopy(all_opt)
            opt['pnp']['lamb'] = lamb
            opt['pnp']['denoisor_pth'] = pnp_opt['denoisor_sigma'][str(denoisor_sigma)]
            opt['pnp']['denoisor_sigma'] = denoisor_sigma
            opts.append(opt)
    return opts

def gen_logger(opt):
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    logger_name = 'pnp'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    return logger

def search_args(json_path='../options/pnp/search_sndncnn.json'):
    opts = gen_opts(json_path)
    logger = gen_logger(opts[0])
    opt_max_psnr = None
    opt_max_ssim = None
    max_psnr = 0.
    max_ssim = 0.
    for opt in opts:
        pnp_model, H_paths, L_paths, noise, n_channels, device = unpack_opt(opt)
        cur_psnr, cur_ssim = eval(pnp_model, H_paths, L_paths, 
                        noise, n_channels, device, logger)
        if cur_psnr > max_psnr:
            max_psnr = cur_psnr
            opt_max_psnr = deepcopy(opt)
        if cur_ssim > max_ssim:
            max_ssim = cur_ssim
            opt_max_ssim = deepcopy(opt)

    logger.info('Best Average PSNR/SSIM(RGB) - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(max_psnr, max_ssim))
    
if __name__ == '__main__':
    search_args()