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
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from func_2d.utils import random_click,vis_image
from skimage import io
import scipy.io as sio
import random
from torchvision.transforms import ToPILImage

class CPM(Dataset):
    def __init__(self, args, cfgs, data_path , mode = 'train',prompt = 'click', plane = False):
        self.data_path = data_path
        if mode == 'train':
            self.image_root = data_path + '/train/Images'
            self.label_root = data_path + '/train/Labels'
        elif mode == 'test':
            self.image_root = data_path + '/test/Images'
            self.label_root = data_path + '/test/Labels'
        self.paths = os.listdir(self.image_root)
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.mask_size = args.out_size

        self.num_mask_per_img = 150

        self.transform = A.Compose(
            [ToTensorV2()], p=1
        )

        self.transform_tmp = A.Compose(
          [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfgs.data.get(mode).transform]
           +[A.Resize(args.image_size,args.image_size,p = 1 )] + [ToTensorV2()], p=1)
        # self.transform = A.Compose(
        #    [getattr(A, tf_dict.pop('type'))(**tf_dict) for tf_dict in cfgs.data.get(mode).transform]
        #     + [ToTensorV2()], p=1)
        self.transform2 = A.Compose([
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=cfgs.prompter.space,
                        pad_width_divisor=cfgs.prompter.space, position="top_left", p=1),
            A.Normalize(),
            A.Resize(args.image_size,args.image_size,p = 1),
            ToTensorV2()
        ], p=1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        """Get the images"""
        path = self.paths[index]

        image_path = os.path.join(self.image_root, path)
        mask_path = os.path.join(self.label_root, path.split('.')[0] + '.mat')

        # raw image and rater images
        img = io.imread(image_path)[..., :3]
        #mask = sio.loadmat(mask_path)["inst_map"].astype(np.int32)
        mask = load_maskfile(mask_path)
        #mask[mask > 0.5] = 1
        #mask[mask <= 0.5] = 0

        # apply transform 数据预处理
        '''
        img_tmp = self.transform2(image=img)['image']
        res = self.transform_tmp(image=img, mask=mask)
        #img = self.transform(Image.open(image_path).convert('RGB'))
        img, mask = list(res.values())
        '''
        img_tmp = self.transform2(image=img)['image']
        res = self.transform(image=img, mask=mask)
        img, mask = list(res.values())
        


        ori_shape = mask.shape[:2]
        inst_map, type_map = mask[..., 0], mask[..., 1]
        unique_pids = np.unique(inst_map)[1:]  # remove zero

        cell_num = len(unique_pids)

        if cell_num:

            chosen_pids = unique_pids
            inst_maps_all = []

            # 8.24全选会爆内存 加一下卡个数
            prompt_points_all = []
            prompt_labels_all = []
            for pid in chosen_pids:
                mask_single_cell = torch.eq(inst_map, pid)

                inst_maps_all.append(mask_single_cell)
                coords = torch.argwhere(mask_single_cell)
                center = coords.float().mean(dim=0)
                center = center.round().long()
                if mask_single_cell[center[0], center[1]] == 0:
                    # 如果中心点在背景上，寻找最近的前景点
                    dists = torch.sqrt(((coords - center) ** 2).sum(dim=1).float())
                    closest_idx = dists.argmin()
                    center = coords[closest_idx]
                pt = center[None, [1, 0]]  # 调整顺序为 [y, x]
                prompt_points_all.append(pt)
                prompt_labels_all.append(type_map[pt[0, 1], pt[0, 0]] - 1)
            # 全部的三个
            prompt_points_all = torch.stack(prompt_points_all, dim=0)
            prompt_labels_all = torch.as_tensor(prompt_labels_all)
            inst_map_all = torch.stack(inst_maps_all, dim=0)

            #随机选的部分
            chosen_pids = np.random.choice(
                unique_pids,
                min(cell_num, self.num_mask_per_img),
                replace=False
            )
            inst_maps_choose = []
            prompt_points_choose = []
            for pid in chosen_pids:
                mask_single_cell = torch.eq(inst_map, pid)

                inst_maps_choose.append(mask_single_cell)
                prompt_points_choose.append(
                    random.choice(
                        torch.argwhere(mask_single_cell)
                    )[None, [1, 0]].float())

            prompt_points_choose = torch.stack(prompt_points_choose, dim=0)
            prompt_labels_choose = torch.ones(prompt_points_choose.squeeze(1).shape[:1])

            inst_map_choose = torch.stack(inst_maps_choose, dim=0)


        else:
            prompt_points_all = torch.empty(0, (self.num_neg_prompt + 1), 2)
            prompt_labels_all = torch.empty(0, (self.num_neg_prompt + 1))
            inst_map_all = torch.empty(0, 256, 256)
            cell_types = torch.empty(0)
        
        binary_tensor = (inst_map_all).to(torch.uint8)
        binary_mask = torch.any(binary_tensor, dim=0).to(torch.uint8)
        
        #vis_image(img.unsqueeze(0),type_map.unsqueeze(0).unsqueeze(0),inst_map.unsqueeze(0).unsqueeze(0), os.path.join('/data/hhb/cell_num_mask.jpg'), reverse=False, points=None)

        if self.mode != 'train':
            return img_tmp,img.to(torch.float32),inst_map, type_map.squeeze(0),prompt_points_all.squeeze(1), prompt_labels_all, binary_mask ,torch.as_tensor(ori_shape),index,path.split('.')[0]

        # return img.to(torch.float32), inst_map_choose.long(), prompt_points_choose, prompt_labels_choose.unsqueeze(-1),prompt_points_all.squeeze(1), prompt_labels_all, min(cell_num, self.num_mask_per_img), binary_mask ,torch.as_tensor(ori_shape)
        # #return img, inst_map_all.long(), prompt_points_all, prompt_labels_all.unsqueeze(1),prompt_points_all.squeeze(1), prompt_labels_all, binary_mask 
        return img.to(torch.float32), inst_map_all.long(), prompt_points_all, prompt_labels_all.unsqueeze(1),prompt_points_all.squeeze(1), prompt_labels_all, cell_num,binary_mask ,torch.as_tensor(ori_shape)


def load_maskfile(mask_path: str):
    inst_map = sio.loadmat(mask_path)['inst_map']
    type_map = (inst_map.copy() > 0).astype(float)

    mask = np.stack([inst_map, type_map], axis=-1)
    return mask