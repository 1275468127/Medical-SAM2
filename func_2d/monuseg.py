""" train and test dataset

author jundewu
"""
import os

import numpy as np
import torch
import cv2
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from func_2d.utils import random_click
import scipy.io as sio

class MONUSEG(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'train',prompt = 'click', plane = False):
        self.data_path = data_path
        if mode == 'train':
            self.image_root = data_path + '/train/images'
            self.label_root = data_path + '/train/labels_png'
        elif mode == 'test':
            self.image_root = data_path + '/test/images'
            self.label_root = data_path + '/test/labels_png'
        self.paths = os.listdir(self.image_root)
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.transform = transform
        self.transform_msk = transform_msk

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        """Get the images"""
        path = self.paths[index]

        image_path = os.path.join(self.image_root, path)
        mask_path = os.path.join(self.label_root, path.split('.')[0] + '.png')

        # raw image and rater images
        img = Image.open(image_path).convert('RGB')
        #mask = sio.loadmat(mask_path)["inst_map"].astype(np.int32)
        mask = Image.open(mask_path).convert('L')

        #mask[mask > 0.5] = 1
        #mask[mask <= 0.5] = 0

        # apply transform 数据预处理
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        # find init click and apply majority vote  从mask生成单点提示
        if self.prompt == 'click':
            point_label_cup, pt_cup = random_click(np.array((mask.mean(axis=0)).squeeze(0)), point_label = 1)

            # # Or use any specific rater as GT
            # point_label_cup, pt_cup = random_click(np.array(multi_rater_cup[0, :, :, :].squeeze(0)), point_label = 1)
            # selected_rater_mask_cup_ori = multi_rater_cup[0,:,:,:]
            # selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 

            # selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
            # selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


        image_meta_dict = {'filename_or_obj':path.split('.')[0]}
        return {
            'image':img,
            #'multi_rater': multi_rater_cup, 
            'p_label': point_label_cup,
            'pt':pt_cup, 
            'mask': mask, 
            #'mask_ori': selected_rater_mask_cup_ori,
            'image_meta_dict':image_meta_dict,
        }


