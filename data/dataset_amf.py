import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util


class DatasetAMF(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for image-to-image mapping.
    # Both "paths_L" and "paths_H" are needed.
    # -----------------------------------------
    # e.g., train denoiser with L and H
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetAMF, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.n_channels = opt['n_channels']
        self.patch_size = self.opt['H_size']

        # ------------------------------------
        # get the path of L/H
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_AMF = util.get_image_paths(opt['dataroot_AMF'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_AMF, 'Error: AMF path is empty.'
        assert self.paths_L, 'Error: L path is empty. Plain dataset assumes both L and H are given!'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        # ------------------------------------
        # get L image
        # ------------------------------------
        L_path = self.paths_L[index]
        img_L = util.imread_uint(L_path, self.n_channels)

        # ------------------------------------
        # get AMF image
        # ------------------------------------
        AMF_path = self.paths_AMF[index]
        img_AMF = util.imread_uint(AMF_path, self.n_channels)        

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            patch_AMF = img_AMF[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_L, patch_H = util.augment_img(patch_L, mode=mode), util.augment_img(patch_H, mode=mode)
            patch_AMF = util.augment_img(patch_AMF, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(patch_L), util.uint2tensor3(patch_H)
            img_AMF = util.uint2tensor3(patch_AMF)

        else:

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_L, img_H = util.uint2tensor3(img_L), util.uint2tensor3(img_H)
            img_AMF = util.uint2tensor3(img_AMF)

        return {'L': img_L, 'H': img_H, 'AMF': img_AMF,
                'L_path': L_path, 'H_path': H_path, 'AMF_path': AMF_path}

    def __len__(self):
        return len(self.paths_H)
