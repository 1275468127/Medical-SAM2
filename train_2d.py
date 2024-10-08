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

    device = torch.device('cuda:' + str(args.gpu_device) if torch.cuda.is_available() else 'cpu')

    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=device, distribution = args.distributed)
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
        refuge_train_dataset = MONUSEG(args, cfgs, args.data_path, mode = 'train')
        refuge_test_dataset = MONUSEG(args, cfgs, args.data_path, mode = 'test')

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
        model1.load_state_dict(ckpt['model1'])
        if 'epoch' in ckpt:
            settings.EPOCH = ckpt['epoch']
        seg_dice1,seg_dice2,seg_aji,seg_aji_p,seg_dq,seg_sq,seg_pq = function.validation_sam(args, cfgs, nice_test_loader, settings.EPOCH, model1, net, cfgs.data.num_classes, cfgs.data.post.iou_threshold, calc_map=True)
        print("dice1:",f"{seg_dice1*100:.2f}" ,end=" ")
        print("dice2:",f"{seg_dice2*100:.2f}" ,end=" ")
        print("aji:",f"{seg_aji*100:.2f}" ,end=" ")
        print("aji_p:",f"{seg_aji_p*100:.2f}" ,end=" ")
        print("dq:",f"{seg_dq*100:.2f}" ,end=" ")
        print("sq:",f"{seg_sq*100:.2f}" ,end=" ")
        print("pq:",f"{seg_pq*100:.2f}" )
        return


    '''begain training'''
    detect_loss = []
    segment_loss = []
    all_loss = []
    dice1 = []
    dice2 = []
    aji = []
    aji_p = []
    dq = []
    sq = []
    pq = []

    best_dice = 0.0
    best_aji = 0.0

    settings.EPOCH = 300
    for epoch in range(settings.EPOCH):
        # training
        net.train()
        time_start = time.time()
        log_info = function.train_sam(args, model1, net, matcher, nice_train_loader, criterion,optimizer, epoch, writer)
        logger.info(f'Train loss: {log_info} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        # 分别保存三个loss曲线在一张图上。
        detect_loss_tmp = log_info['loss_reg']+log_info['loss_cls']+log_info['loss_mask']
        # segment_loss_tmp = log_info['loss_focal']+log_info['loss_dice']+log_info['loss_iou']+log_info['loss_dice_semantic']
        segment_loss_tmp = log_info['loss_focal']+log_info['loss_dice']+log_info['loss_iou']
        #segment_loss_tmp = log_info['loss_focal']+log_info['loss_sam2']
        all_loss_tmp = detect_loss_tmp+segment_loss_tmp
        detect_loss.append(detect_loss_tmp)
        segment_loss.append(segment_loss_tmp)
        all_loss.append(all_loss_tmp)

        # validation
        
        net.eval()
        if epoch % args.val_freq == 0 or epoch == settings.EPOCH-1:

            seg_dice1,seg_dice2,seg_aji,seg_aji_p,seg_dq,seg_sq,seg_pq = function.validation_sam(args, cfgs, nice_test_loader, epoch, model1, net, cfgs.data.num_classes, cfgs.data.post.iou_threshold, calc_map=True)
            print("dice1:",f"{seg_dice1*100:.2f}" ,end=" ")
            print("dice2:",f"{seg_dice2*100:.2f}" ,end=" ")
            print("aji:",f"{seg_aji*100:.2f}" ,end=" ")
            print("aji_p:",f"{seg_aji_p*100:.2f}" ,end=" ")
            print("dq:",f"{seg_dq*100:.2f}" ,end=" ")
            print("sq:",f"{seg_sq*100:.2f}" ,end=" ")
            print("pq:",f"{seg_pq*100:.2f}" )
            dice1.append(seg_dice1)
            dice2.append(seg_dice2)
            aji.append(seg_aji)
            aji_p.append(seg_aji_p)
            dq.append(seg_dq)
            sq.append(seg_sq)
            pq.append(seg_pq)

            if seg_dice1 > best_dice:
                best_dice = seg_dice1
                torch.save({'model': net.state_dict(), 'model1': model1.state_dict(), 'parameter': net._parameters, 'epoch': epoch}, os.path.join(args.path_helper['ckpt_path'], 'base_dice_epoch.pth'))
        
            if seg_aji > best_aji:
                best_aji = seg_aji
                torch.save({'model': net.state_dict(), 'model1': model1.state_dict(), 'parameter': net._parameters, 'epoch': epoch}, os.path.join(args.path_helper['ckpt_path'], 'base_aji_epoch.pth'))

    writer.close()

    # 绘制损失曲线
    epochs = np.arange(1, len(detect_loss) + 1)

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(20, 12))

    # 绘制图表（包含 3 条线）
    ax1.plot(epochs, detect_loss, marker='o', linestyle='-', color='b', label='Detect Loss (normalized)')
    ax1.plot(epochs, segment_loss, marker='o', linestyle='-', color='g', label='Segment Loss (normalized)')
    ax1.plot(epochs, all_loss, marker='o', linestyle='-', color='r', label='Total Loss (normalized)')

    # 设置图表标题和标签
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss/metrics')
    ax1.set_title('Training Losses and Metrics (Normalized)')
    ax1.grid(True)
    ax1.legend()

    # 调整布局
    plt.tight_layout()

    # 保存图像到指定位置
    save_path = os.path.join(args.path_helper['sample_path'], 'Loss.png')
    plt.savefig(save_path)
    plt.close()  # 关闭图形，避免显示图形界面


    # 创建图表
    epochs = np.arange(1, len(dice1) + 1)
    fig, ax2 = plt.subplots(figsize=(20, 12))

    # 绘制图表，包含 6 条线
    ax2.plot(epochs, dice1, marker='o', linestyle='-', color='b', label='Dice Score')
    ax2.plot(epochs, dice2, marker='o', linestyle='-', color='b', label='Dice Score')
    ax2.plot(epochs, aji, marker='o', linestyle='-', color='g', label='AJI')
    ax2.plot(epochs, aji_p, marker='o', linestyle='-', color='r', label='AJI Plus')
    ax2.plot(epochs, dq, marker='o', linestyle='-', color='c', label='DQ (Detection Quality)')
    ax2.plot(epochs, sq, marker='o', linestyle='-', color='m', label='SQ (Segmentation Quality)')
    ax2.plot(epochs, pq, marker='o', linestyle='-', color='y', label='PQ (Panoptic Quality)')

    # 设置图表标题和标签
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Evaluation Metrics Over Epochs')
    ax2.grid(True)
    ax2.legend()

    # 调整布局
    plt.tight_layout()

    # 保存评价指标图像到指定位置
    save_path = os.path.join(args.path_helper['sample_path'], 'Metrics.png')
    plt.savefig(save_path)
    plt.close()  # 关闭图形，避免显示图形界面

if __name__ == '__main__':
    main()