import argparse

#先cpm后monuseg再cpm
def parse_args():    
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('-print_freq', type=int, default=100, help='print_freq')
    parser.add_argument(
        "--model_ema_steps",
        type=int,
        default=1,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99)",
    )
    parser.add_argument('--clip-grad', type=float, default=0.1,
                        help='Clip gradient norm (default: 0.1)')
    parser.add_argument("--overlap", default=64, type=int, help="overlapping pixels")
    parser.add_argument("--crop_size", default=256, type=int, help="overlapping pixels")
    #parser.add_argument('-eval', type=str, default='', help='net type')
    parser.add_argument('--eval', action='store_true')
    
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
    parser.add_argument('-exp_name', default='samba_train_test', type=str, help='experiment name')
    parser.add_argument('-vis', type=bool, default=True, help='Generate visualisation during validation')
    parser.add_argument('-train_vis', type=bool, default=False, help='Generate visualisation during training')
    parser.add_argument('-prompt', type=str, default='click', help='type of prompt, bbox or click')
    parser.add_argument('-prompt_freq', type=int, default=2, help='frequency of giving prompt in 3D images')
    parser.add_argument('-pretrain', type=str, default=None, help='path of pretrain weights')
    parser.add_argument('-val_freq',type=int,default=3,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-image_size', type=int, default=256, help='image_size')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default="cpm",type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', type=str, default="/data/hhb/project/MedSAM2-inst/checkpoints/sam2_hiera_small.pt" , help='sam checkpoint address')
    #parser.add_argument('-sam_ckpt', type=str, default="/data/hhb/project1/PS-SAM2/Medical-SAM2/logs/samba_train_test_2024_09_14_21_18_04/Model/latest_epoch.pth" , help='sam checkpoint address')
    parser.add_argument('-sam_config', type=str, default="sam2_hiera_s" , help='sam checkpoint address')
    parser.add_argument('-video_length', type=int, default=2, help='sam checkpoint address')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    #parser.add_argument('-weights', type=str, default ="/data/hhb/project/Medical-SAM2/logs/samba_train_test_2024_08_18_19_26_41/Model/best_dice_checkpoint.pth", help='the weights file you want to test')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation')
    parser.add_argument('-memory_bank_size', type=int, default=16, help='sam 2d memory bank size')
    parser.add_argument(
    '-data_path',
    type=str,
    default='/data/hhb/data/cpm17_256',
    help='The path of segmentation data') 
    opt = parser.parse_args()

    return opt
