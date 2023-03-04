import logging
import os.path
from collections import OrderedDict
import sys
from torch.utils.data import DataLoader
sys.path.append("..") 

from utils import utils_option as option
from utils import utils_image as util
from utils import utils_logger

from data.select_dataset import define_Dataset

def gen_logger(opt):
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    logger_name = 'pnp'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    return logger

def get_opt(json_path):
    opt = option.parse(json_path, is_train=False)
    opt = option.dict_to_nonedict(opt)
    return opt

def save_opt(opt, pth='sndncnn.json'):
    import json
    with open(pth, 'w') as f:
        json.dump(opt, f)

def get_test_loader(opt):
    dataset_opt = opt['datasets']['test']
    dataset_opt['sigma_test'] = opt['sigma']
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    return test_loader

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