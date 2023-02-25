import torch
from torch import nn
from collections import OrderedDict

def select_denoisor(opt):
    opt_net = opt['netG']
    denoisor_type = opt_net['denoisor']
    denoisor_len = opt_net['admm_iter_num']
    if denoisor_type == 'sndncnn_single':
        denoisor = Denoisor_SNDnCNN_Single()
    elif denoisor_type == 'sndncnn':
        denoisor = Denoisor_SNDnCNN(denoisor_len)
    elif denoisor_type == 'dncnn':
        denoisor = Denoisor_DnCNN(denoisor_len)
    return denoisor

class Denoisor(nn.Module):
    def __init__(self):
        super(Denoisor, self).__init__()
        self.denoisor = None

    def load(self, pth, max_load_len):
        pass

    def get_denoisor(self, i):
        pass

    def save(self, pth):
        pass

class Denoisor_SNDnCNN_Single(Denoisor):
    def __init__(self, len_denoisor=-1):
        super(Denoisor_SNDnCNN_Single, self).__init__()
        from models.network_sndncnn import SNDnCNN as Net
        self.denoisor = Net(channels=1, lip=-1)
        print("init Denoisor_SNDnCNN_Single")

    def load(self, pth, max_load_len=-1):
        denoisor = self.get_denoisor()
        d = torch.load(pth)
        d = self._sndncnn_to_dncnn(d)
        denoisor.load_state_dict(d)

    def get_denoisor(self, i=-1):
        return self.denoisor

    def _sndncnn_to_dncnn(self, d):
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

class Denoisor_SNDnCNN(Denoisor):
    def __init__(self, len_denoisor):
        super(Denoisor_SNDnCNN, self).__init__()
        self.len_denoisor = len_denoisor
        from models.network_sndncnn import SNDnCNN as Net
        for i in range(len_denoisor):
            setattr(self, 'denoisor' + str(i), 
                Net(channels=1, lip=-1)
            )

    def load(self, pth, max_load_len):
        for i in range(min(self.len_denoisor, max_load_len)):
            denoisor_i = self.get_denoisor(i)
            denoisor_i.load_state_dict(torch.load(pth))

    def get_denoisor(self, i):
        return getattr(self, 'denoisor' + str(i))

def _load_dncnn(model, pth):
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

class Denoisor_DnCNN(Denoisor):
    def __init__(self, len_denoisor):
        super(Denoisor_DnCNN, self).__init__()
        self.len_denoisor = len_denoisor
        from models.network_dncnn import DnCNN as Net
        for i in range(len_denoisor):
            setattr(self, 'denoisor' + str(i), 
                Net(in_nc=1, out_nc=1, nc=64, nb=17, act_mode='R')
            )

    def load(self, pth, max_load_len):
        for i in range(min(self.len_denoisor, max_load_len)):
            denoisor_i = self.get_denoisor(i)
            _load_dncnn(denoisor_i, pth)

    def get_denoisor(self, i):
        return getattr(self, 'denoisor' + str(i))