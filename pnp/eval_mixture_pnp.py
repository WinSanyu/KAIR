import argparse
import sys
sys.path.append("..") 

from models.select_model import define_Model
from pnp.util_pnp import get_test_loader, eval, get_opt, gen_logger

def unpack_opt(opt):
    opt['datasets']['test']['sigma'] = opt['sigma']
    opt['datasets']['test']['sigma_test'] = opt['sigma']
    opt['datasets']['test']['sp'] = opt['sp']
    test_loader = get_test_loader(opt)
    model = define_Model(opt)
    return model, test_loader

def _get_json_path():
    json_path = '../options/pnp/mixture.json'
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    json_path = parser.parse_args().opt
    return json_path

json_path = _get_json_path()
opt = get_opt(json_path)
logger = gen_logger(opt)
eval(*unpack_opt(opt), logger)

