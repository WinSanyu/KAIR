import os.path
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util


class DatasetMixture(data.Dataset):
    """
    # -----------------------------------------
    # Get L/H for denosing on AWGN with fixed sigma.
    # Only dataroot_H is needed.
    # -----------------------------------------
    # e.g., DnCNN
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetMixture, self).__init__()
        print('Dataset: Denosing on AWGN and SPIN with fixed sigma. Only dataroot_H is needed.')
        self.opt = opt
        self.n_channels = opt['n_channels']
        self.patch_size = opt['H_size']
        self.sigma = opt['sigma']
        self.sigma_test = self.sigma
        self.sp = opt['sp']

        # ------------------------------------
        # get path of H
        # return None if input is None
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])

    def __getitem__(self, index):

        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]
        img_H = util.imread_uint(H_path, self.n_channels)

        L_path = H_path

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # get L/H patch pairs
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # randomly crop the patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # augmentation - flip, rotate
            # --------------------------------
            mode = random.randint(0, 7)
            patch_H = util.augment_img(patch_H, mode=mode)

            # --------------------------------
            # HWC to CHW, numpy(uint) to tensor
            # --------------------------------
            img_H = util.uint2tensor3(patch_H)
            img_L = img_H.clone()

            # --------------------------------
            # add noise
            # --------------------------------
            noise = torch.randn(img_L.size()).mul_(self.sigma/255.0)
            img_L.add_(noise)

            mask = torch.rand(img_L.shape)
            mask1 = mask <= self.sp/2
            mask2 = mask >= 1- self.sp/2
            img_L[mask1] = 0.
            img_L[mask2] = 1.

        else:
            """
            # --------------------------------
            # get L/H image pairs
            # --------------------------------
            """
            img_H = util.uint2single(img_H)
            img_L = np.copy(img_H)

            # --------------------------------
            # add noise
            # --------------------------------
            np.random.seed(seed=0)
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)
            mask = np.random.rand(*img_L.shape)
            mask1 = mask <= self.sp/2
            mask2 = mask >= 1- self.sp/2
            img_L[mask1] = 0.
            img_L[mask2] = 1.

            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_L = util.single2tensor3(img_L)
            img_H = util.single2tensor3(img_H)

        return {'L': img_L, 'H': img_H, 'H_path': H_path, 'L_path': L_path}

    def __len__(self):
        return len(self.paths_H)
