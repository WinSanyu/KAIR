from copy import deepcopy
import sys
sys.path.append("..") 

from pnp.eval_cpnp import unpack_opt, eval
from pnp.util_pnp import get_opt, gen_logger, save_opt

def gen_opts(json_path):
    '''Generate all opt grids to be searched'''
    opts = []
    all_opt = get_opt(json_path)
    pnp_opt = all_opt['pnp']
    for denoisor_sigma in pnp_opt['denoisor_sigma']:
        for lamb in range(*pnp_opt['lamb']):
            for irl1_iter_num in range(*pnp_opt['irl1_iter_num']):
                opt = deepcopy(all_opt)
                opt['netG']['lamb'] = lamb
                opt['netG']['admm_iter_num'] = opt['pnp']['admm_iter_num']
                opt['netG']['denoisor_pth'] = pnp_opt['denoisor_sigma'][str(denoisor_sigma)]
                opt['netG']['denoisor_sigma'] = denoisor_sigma
                opt['netG']['irl1_iter_num'] = irl1_iter_num
                opt['netG']['mu'] = opt['pnp']['mu']
                opt['netG']['rho'] = opt['pnp']['rho']
                opt['netG']['eps'] = opt['pnp']['eps']
                opts.append(opt)
    return opts

def search_args(json_path):
    opts = gen_opts(json_path)
    logger = gen_logger(opts[0])
    opt_max_psnr = None
    opt_max_ssim = None
    max_psnr = 0.
    max_ssim = 0.
    for opt in opts:
        logger.info("denoisor: {}, lamb: {}, admm: {}, irl1: {}".format(
                    opt['pnp']['denoisor_sigma'], opt['pnp']['lamb'], 
                    opt['pnp']['admm_iter_num'], opt['pnp']['irl1_iter_num']))
        
        cur_psnr, cur_ssim = eval(*unpack_opt(opt), logger)
        if cur_psnr > max_psnr:
            max_psnr = cur_psnr
            opt_max_psnr = deepcopy(opt)
        if cur_ssim > max_ssim:
            max_ssim = cur_ssim
            opt_max_ssim = deepcopy(opt)

    logger.info('Best Average PSNR/SSIM - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(max_psnr, max_ssim))
    save_opt(opt_max_psnr, 'cpnp_psnr.json')
    save_opt(opt_max_ssim, 'cpnp_ssim.json')
    
if __name__ == '__main__':
    import argparse
    json_path = '../options/pnp/search_sndncnn.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    json_path = parser.parse_args().opt
    search_args(json_path)