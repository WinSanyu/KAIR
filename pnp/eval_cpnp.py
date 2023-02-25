import torch
import torch.nn as nn
import numpy as np
import sys
import os.path
from collections import OrderedDict
sys.path.append("..") 

from utils import utils_image as util
from models.select_model import define_Model
from pnp.util_pnp import get_test_loader

def eval(model, test_loader, logger):
    idx = 0
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    for test_data in test_loader:
        idx += 1

        image_name_ext = os.path.basename(test_data['L_path'][0])

        model.feed_data(test_data)
        model.test()

        visuals = model.current_visuals()
        img_E = util.tensor2uint(visuals['E'])
        img_H = util.tensor2uint(visuals['H'])

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
    test_loader = get_test_loader(opt)
    opt['netG']['sigma'] = opt['sigma']
    model = define_Model(opt)
    return model, test_loader

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
