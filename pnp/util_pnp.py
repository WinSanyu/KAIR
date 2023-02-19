import torch
import logging
import os.path
import sys
from collections import OrderedDict
sys.path.append("..") 

from utils import utils_option as option
from utils import utils_image as util
from utils import utils_logger

# from data.select_dataset import define_Dataset
from models.select_network import define_G

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

def load_dncnn(model, pth):
    old_para_dict = torch.load(pth)
    para_dict = OrderedDict()
    i = 0
    for old_key in old_para_dict:
        new_key = 'model.' + str(i//2*2)
        if i % 2:
            new_key += '.bias'
        else:
            new_key += '.weight'
        para_dict[new_key] = old_para_dict[old_key]
        # print(new_key, "  <====  ", old_key)
        i += 1
    model.load_state_dict(para_dict, strict=True)

def sndncnn_to_dncnn(d):
    from collections import OrderedDict
    res = OrderedDict()
    for key in d:
        if "_u" in key:
            continue
        if "_orig" in key:
            continue

        val = d[key]
        dncnn_key = key
        res[dncnn_key] = val
    return res

def get_network_eval(opt):
    model = define_G(opt)
    model_path = opt['pnp']['denoisor_pth'] # opt['path']['pretrained_netG']
    model_weights = sndncnn_to_dncnn(torch.load(model_path))
    model.load_state_dict(model_weights, strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model

def save_opt(opt, pth='sndncnn.json'):
    import json
    with open(pth, 'w') as f:
        json.dump(opt, f)


# def get_test_loader(opt):
#     dataset_opt = opt['datasets']['test']
#     dataset_opt['sigma_test'] = opt['sigma_test']
#     test_set = define_Dataset(dataset_opt)
#     test_loader = DataLoader(test_set, batch_size=1,
#                              shuffle=False, num_workers=1,
#                              drop_last=False, pin_memory=True)
#     return test_loader