""" helper function

author junde
"""

import logging
import math
import os
import pathlib
import random
import shutil
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from typing import BinaryIO, List, Optional, Text, Tuple, Union

import dateutil.tz
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
from PIL import Image
from torch.autograd import Function


import cfg

import pandas as pd


args = cfg.parse_args()

device = torch.device('cuda:' + str(args.gpu_device) if torch.cuda.is_available() else 'cpu')



def get_network(args, net, use_gpu=True, gpu_device = 0, distribution = True):
    """ return given network
    """


    if net == 'sam2':
        from sam2_train.build_sam import build_sam2
        from sam2_train.sam2_image_predictor import SAM2ImagePredictor
        '''
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        '''
        net = build_sam2(args.sam_config, args.sam_ckpt, device=device)


    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if use_gpu:
        #net = net.cuda(device = gpu_device)
        if distribution != 'none':
            net = torch.nn.DataParallel(net,device_ids=[int(id) for id in args.distributed.split(',')])
            net = net.to(device=gpu_device)
        else:
            net = net.to(device=gpu_device)

    return net


@torch.no_grad()
def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    **kwargs
) -> torch.Tensor:
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clamp(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


@torch.no_grad()
def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[Text, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)
    

def create_logger(log_dir, phase='train'):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, phase)
    final_log_file = os.path.join(log_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def set_log_dir(root_dir, exp_name):
    path_dict = {}
    os.makedirs(root_dir, exist_ok=True)

    # set log path
    exp_path = os.path.join(root_dir, exp_name)
    now = datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    prefix = exp_path + '_' + timestamp
    os.makedirs(prefix)
    path_dict['prefix'] = prefix

    # set checkpoint path
    ckpt_path = os.path.join(prefix, 'Model')
    os.makedirs(ckpt_path)
    path_dict['ckpt_path'] = ckpt_path

    log_path = os.path.join(prefix, 'Log')
    os.makedirs(log_path)
    path_dict['log_path'] = log_path

    # set sample image path for fid calculation
    sample_path = os.path.join(prefix, 'Samples')
    os.makedirs(sample_path)
    path_dict['sample_path'] = sample_path

    return path_dict


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best:
        torch.save(states, os.path.join(output_dir, 'checkpoint_best.pth'))



def iou(outputs: np.array, labels: np.array):
    
    SMOOTH = 1e-6
    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)


    return iou.mean()

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).to(device = input.device).zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)




def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


import cv2
def visualize_points_on_images(mask, points, save_path):
    # 获取尺寸
    b, c, h, w = mask.shape
    
    # 将 mask 转换为 CPU 张量并转换为 NumPy 数组
    mask_val = mask[0, 0].cpu().numpy()

    # 创建一个空白的 RGB 图像
    result_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 使用 NumPy 的布尔索引处理 mask
    result_img[mask_val == 0] = [0, 0, 0]  # 黑色
    result_img[mask_val > 0] = [255, 255, 255]  # 白色

    # 获取并处理 points
    points_int = np.round(points[0].detach().cpu().numpy()).astype(int)
    radius = 3  # 点的半径

    # 绘制绿色的点（使用 OpenCV 的 circle 函数）
    for point in points_int:
        y, x = point
        cv2.circle(result_img, (x, y), radius, (0, 255, 0), thickness=-1)

    # 保存结果图像
    cv2.imwrite(save_path, result_img)

def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random()*255 for i in range(3)]
    return r, g, b

def inst_image(pred_masks, save_path, reverse = False, points = None):
    pred_labeled = pred_masks.astype(np.uint8)
    pred_colored = np.zeros((pred_masks.shape[0], pred_masks.shape[1], 3))
    pred_labeled_cnum = pred_labeled.max() + 1
    for k in range(1, pred_labeled_cnum):
        pred_colored[pred_labeled == k, :] = np.array(get_random_color())
    filename = save_path
    print('pred_labeled color number = %d' % (pred_labeled_cnum))
    # misc.imsave(filename, pred_colored)
    cv2.imwrite(filename, pred_colored)

def get_inst_image(pred_masks): # 输入1，3，256，256
    pred_labeled = pred_masks.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_colored = np.zeros((pred_masks.shape[2], pred_masks.shape[3], 3))
    pred_labeled_cnum = pred_labeled.max() + 1
    for k in range(1, pred_labeled_cnum):
        pred_colored[pred_labeled == k, :] = np.array(get_random_color())
    return torch.tensor(pred_colored).unsqueeze(0).permute(0, 3, 1, 2)

def vis_inst_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None):
    
    b,c,h,w = pred_masks.size()
    row_num = min(b, 4)
    
    imgs = torchvision.transforms.Resize((h,w))(imgs)
    if imgs.size(1) == 1:
        imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)

    #imgs = imgs / 255.0  # 归一化到 [0, 1]
    pred_masks = get_inst_image(pred_masks).to(device)
    gt_masks = get_inst_image(gt_masks).to(device)
    pred_masks = pred_masks / 255.0
    gt_masks = gt_masks / 255.0
    tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
    # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
    compose = torch.cat(tup,0)
    vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)

    return

def compare_and_color(pred_map, true_map):
    b, c, h, w = pred_map.shape
    result_img = np.zeros((h, w, 3), dtype=np.uint8)

    pred_val = pred_map[0, 0].cpu().numpy()
    true_val = true_map[0, 0].cpu().numpy()

    # 条件判断并赋值，利用NumPy的布尔索引进行矢量化
    mask_black = (pred_val == 0) & (true_val == 0)
    mask_red = (pred_val > 0) & (true_val == 0)
    mask_blue = (pred_val == 0) & (true_val > 0)
    mask_green = (pred_val > 0) & (true_val > 0)

    # 直接赋值
    result_img[mask_black] = [0, 0, 0]    # 黑色
    result_img[mask_red] = [0, 0, 255]    # 红色
    result_img[mask_blue] = [255, 0, 0]   # 蓝色
    result_img[mask_green] = [0, 255, 0]  # 绿色

    return result_img


def vis_compare_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None):
    
    b,c,h,w = pred_masks.size()
    row_num = min(b, 4)
    
    imgs = torchvision.transforms.Resize((h,w))(imgs)
    if imgs.size(1) == 1:
        imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)

    imgs = imgs / 255.0  # 归一化到 [0, 1]
    
    result_img = compare_and_color(pred_masks, gt_masks)

    cv2.imwrite(save_path, result_img)

    return

def vis_compare_point(imgs, pred_masks, gt_masks, points, save_path, reverse = False):
    
    b,c,h,w = pred_masks.size()
    row_num = min(b, 4)
    
    imgs = torchvision.transforms.Resize((h,w))(imgs)
    if imgs.size(1) == 1:
        imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)

    imgs = imgs / 255.0  # 归一化到 [0, 1]
    
    result_img = compare_and_color(pred_masks, gt_masks)

    radius = 3  # 点的半径
    # 绘制绿色的点（使用 OpenCV 的 circle 函数）
    for point in points:
        x, y = point
        x = int(x)
        y = int(y)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(result_img, (x, y), radius, (256, 256, 256), thickness=-1)

    cv2.imwrite(save_path, result_img)

    return

def vis_image(imgs, pred_masks, gt_masks, save_path, reverse = False, points = None):
    
    b,c,h,w = pred_masks.size()
    dev = pred_masks.get_device()
    row_num = min(b, 4)
    
    if torch.max(pred_masks) > 1 :
        pred_masks = pred_masks > 0.5

    if torch.max(pred_masks) > 1 or torch.min(pred_masks) < 0:
        pred_masks = torch.sigmoid(pred_masks)

    if reverse == True:
        pred_masks = 1 - pred_masks
        gt_masks = 1 - gt_masks
    if c == 2: # for REFUGE multi mask output
        pred_disc, pred_cup = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), pred_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_disc, gt_cup = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w), gt_masks[:,1,:,:].unsqueeze(1).expand(b,3,h,w)
        tup = (imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:])
        compose = torch.cat(tup, 0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    elif c > 2: # for multi-class segmentation > 2 classes
        preds = []
        gts = []
        for i in range(0, c):
            pred = pred_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            preds.append(pred)
            gt = gt_masks[:,i,:,:].unsqueeze(1).expand(b,3,h,w)
            gts.append(gt)
        tup = [imgs[:row_num,:,:,:]] + preds + gts
        compose = torch.cat(tup,0)
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)
    else:
        imgs = torchvision.transforms.Resize((h,w))(imgs)
        if imgs.size(1) == 1:
            imgs = imgs[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        pred_masks = pred_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        gt_masks = gt_masks[:,0,:,:].unsqueeze(1).expand(b,3,h,w)
        if points != None:
            for i in range(b):
                
                points_np = points.detach().cpu().numpy()  # 分离计算图，移到 CPU，并转换为 NumPy 数组

                # 执行转换
                p = np.round(points_np / args.image_size * args.out_size).astype(int)
                #p = np.round(points.cpu()/args.image_size * args.out_size).to(dtype = torch.int)
                
                gt_masks[i,0,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.5
                gt_masks[i,1,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.1
                gt_masks[i,2,p[i,0]-2:p[i,0]+2,p[i,1]-2:p[i,1]+2] = 0.4
                # gt_masks[i,0,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.5
                # gt_masks[i,1,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.1
                # gt_masks[i,2,p[i,0]-5:p[i,0]+5,p[i,1]-5:p[i,1]+5] = 0.4

        imgs = imgs / 255.0  # 归一化到 [0, 1]
        tup = (imgs[:row_num,:,:,:],pred_masks[:row_num,:,:,:], gt_masks[:row_num,:,:,:])
        # compose = torch.cat((imgs[:row_num,:,:,:],pred_disc[:row_num,:,:,:], pred_cup[:row_num,:,:,:], gt_disc[:row_num,:,:,:], gt_cup[:row_num,:,:,:]),0)
        compose = torch.cat(tup,0)  
        vutils.save_image(compose, fp = save_path, nrow = row_num, padding = 10)

    return

def eval_seg(pred,true_mask_p,threshold):
    '''
    threshold: a int or a tuple of int
    masks: [b,2,h,w]
    pred: [b,2,h,w]
    '''
    b, c, h, w = pred.size()
    if c == 2:
        iou_d, iou_c, disc_dice, cup_dice = 0,0,0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')
            cup_pred = vpred_cpu[:,1,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
            cup_mask = gt_vmask_p [:, 1, :, :].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            iou_d += iou(disc_pred,disc_mask)
            iou_c += iou(cup_pred,cup_mask)

            '''dice for torch'''
            disc_dice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            cup_dice += dice_coeff(vpred[:,1,:,:], gt_vmask_p[:,1,:,:]).item()
            
        return iou_d / len(threshold), iou_c / len(threshold), disc_dice / len(threshold), cup_dice / len(threshold)
    elif c > 2: # for multi-class segmentation > 2 classes
        ious = [0] * c
        dices = [0] * c
        for th in threshold:
            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            for i in range(0, c):
                pred = vpred_cpu[:,i,:,:].numpy().astype('int32')
                mask = gt_vmask_p[:,i,:,:].squeeze(1).cpu().numpy().astype('int32')
        
                '''iou for numpy'''
                ious[i] += iou(pred,mask)

                '''dice for torch'''
                dices[i] += dice_coeff(vpred[:,i,:,:], gt_vmask_p[:,i,:,:]).item()
            
        return tuple(np.array(ious + dices) / len(threshold)) # tuple has a total number of c * 2
    else:
        eiou, edice = 0,0
        for th in threshold:

            gt_vmask_p = (true_mask_p > th).float()
            vpred = (pred > th).float()
            vpred_cpu = vpred.cpu()
            disc_pred = vpred_cpu[:,0,:,:].numpy().astype('int32')

            disc_mask = gt_vmask_p [:,0,:,:].squeeze(1).cpu().numpy().astype('int32')
    
            '''iou for numpy'''
            eiou += iou(disc_pred,disc_mask)

            '''dice for torch'''
            edice += dice_coeff(vpred[:,0,:,:], gt_vmask_p[:,0,:,:]).item()
            
        return eiou / len(threshold), edice / len(threshold)


def random_click(mask, point_label = 1):
    max_label = max(set(mask.flatten()))
    if round(max_label) == 0:
        point_label = round(max_label)
    indices = np.argwhere(mask == max_label) 
    return point_label, indices[np.random.randint(len(indices))]

def agree_click(mask, label = 1):
    # max agreement position
    indices = np.argwhere(mask == label) 
    if len(indices) == 0:
        label = 1 - label
        indices = np.argwhere(mask == label) 
    return label, indices[np.random.randint(len(indices))]


def random_box(multi_rater):
    max_value = torch.max(multi_rater[:,0,:,:], dim=0)[0]
    max_value_position = torch.nonzero(max_value)

    x_coords = max_value_position[:, 0]
    y_coords = max_value_position[:, 1]


    x_min = int(torch.min(x_coords))
    x_max = int(torch.max(x_coords))
    y_min = int(torch.min(y_coords))
    y_max = int(torch.max(y_coords))


    x_min = random.choice(np.arange(x_min-10,x_min+11))
    x_max = random.choice(np.arange(x_max-10,x_max+11))
    y_min = random.choice(np.arange(y_min-10,y_min+11))
    y_max = random.choice(np.arange(y_max-10,y_max+11))

    return x_min, x_max, y_min, y_max


