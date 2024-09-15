# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Jiayuan Zhu
"""

import os
import time

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from dataset import *
from torch.utils.data import DataLoader

import cfg
import func_2d.function as function
from conf import settings
#from models.discriminatorlayer import discriminator
from func_2d.dataset import *
from func_2d.utils import *
from func_2d.monuseg import *
from func_2d.cpm import *
from mmengine.config import Config
from sam2_train.modeling.dpa_p2pnet import build_model 
from sam2_train.modeling.utils import *
from sam2_train.modeling.criterion import build_criterion

def main():
    cfgs = Config.fromfile(f'/data/hhb/project/Medical-SAM2/args.py')
    args = cfg.parse_args()
    GPUdevice = torch.device('cuda', args.gpu_device)

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    model1 = build_model(cfgs).to(device)

    # optimisation
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    actual_lr = cfgs.optimizer.lr * (cfgs.data.batch_size_per_gpu * get_world_size()) / 8  # linear scaling rule
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, list(model1.parameters()) + list(net.parameters())),
        lr=actual_lr,
        weight_decay=cfgs.optimizer.weight_decay
    )
    criterion , matcher = build_criterion(cfgs, device)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

    '''load pretrained model'''

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)


    '''segmentation data'''
    transform_train = transforms.Compose([
        transforms.Resize((args.image_size,args.image_size)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    
    # example of REFUGE dataset
    if args.dataset == 'monuseg':
        '''REFUGE data'''
        refuge_train_dataset = MONUSEG(args, args.data_path, transform = transform_train, mode = 'train')
        refuge_test_dataset = MONUSEG(args, args.data_path, transform = transform_test, mode = 'test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=True, num_workers=2, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
    elif args.dataset == 'cpm':
        '''REFUGE data'''
        refuge_train_dataset = CPM(args, cfgs, args.data_path, mode = 'train')
        refuge_test_dataset = CPM(args, cfgs, args.data_path, mode = 'test')

        nice_train_loader = DataLoader(refuge_train_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
        nice_test_loader = DataLoader(refuge_test_dataset, batch_size=args.b, shuffle=False, num_workers=2, pin_memory=True)
        '''end'''


    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    if args.eval:
        ckpt = torch.load(args.sam_ckpt, map_location="cpu")
        #model1.load_state_dict(ckpt['model1'])
        eiou, edice = function.validation_sam(args, cfgs, nice_test_loader, settings.EPOCH, model1, net, cfgs.data.num_classes, cfgs.data.post.iou_threshold, calc_map=True)
        logger.info(f'IOU: {eiou}, DICE: {edice} || @ epoch {settings.EPOCH}.')

    '''begain training'''
    best_tol = 1e4
    best_dice = 0.0

    settings.EPOCH = 1000
    for epoch in range(settings.EPOCH):
        # training
        net.train()
        time_start = time.time()
        loss = function.train_sam(args, model1, net, matcher, nice_train_loader, criterion,optimizer, epoch, writer)
        logger.info(f'Train loss: {loss} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        # validation
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:

            seg_dice,seg_aji,seg_aji_p,seg_dq,seg_sq,seg_pq = function.validation_sam(args, cfgs, nice_test_loader, epoch, model1, net, cfgs.data.num_classes, cfgs.data.post.iou_threshold, calc_map=True)
            print("dice:",f"{seg_dice*100:.2f}" ,end=" ")
            print("aji:",f"{seg_aji*100:.2f}" ,end=" ")
            print("aji_p:",f"{seg_aji_p*100:.2f}" ,end=" ")
            print("dq:",f"{seg_dq*100:.2f}" ,end=" ")
            print("sq:",f"{seg_sq*100:.2f}" ,end=" ")
            print("pq:",f"{seg_pq*100:.2f}" )

            if seg_dice > best_dice:
                best_dice = seg_dice
                torch.save({'model': net.state_dict(), 'model1': model1.state_dict(), 'parameter': net._parameters}, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))


    writer.close()


if __name__ == '__main__':
    main()