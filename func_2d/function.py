
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import cfg
from conf import settings
from func_2d.utils import *
import pandas as pd
from sam2_train.modeling.stats_utils import *
from sam2_train.modeling.utils import *
from func_2d.eval_map import eval_map
import prettytable as pt

args = cfg.parse_args()

GPUdevice = torch.device('cuda', args.gpu_device)
pos_weight = torch.ones([1]).cuda(device=GPUdevice)*2
criterion_G = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
mask_type = torch.float32

torch.backends.cudnn.benchmark = True


def train_sam(args, point_net, net, matcher, train_loader, criterion,optimizer, epoch, writer):
    # train mode
    point_net.train()
    net.train()
    optimizer.zero_grad()
    criterion.train()

    log_info = dict()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # init
    epoch_loss = 0
    memory_bank_list = []
    lossfunc = criterion_G
    #feat_sizes = [(256, 256), (128, 128), (64, 64)]
    feat_sizes = [(64, 64), (32, 32), (16, 16)]


    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for data_iter_step, (images, inst_masks,points_choose,labels_choose, points_list, labels_list , cell_nums , masks, ori_shape) in enumerate(
            metric_logger.log_every(train_loader, args.print_freq, header)):\
            
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            imgs = images.to(device) # cpm:torch.Size([8, 3, 512, 512])   pannuke:torch.Size([16, 3, 256, 256])
            masks = masks.to(device).float()
            targets = {
                'gt_masks': masks,
                'gt_nums': [len(points) for points in points_list],
                'gt_points': [points.view(-1, 2).to(device).float() for points in points_list],
                'gt_labels': [labels.to(device).long() for labels in labels_list],
            }
            
            outputs1,feats_origin,image_embedding,feats = point_net(imgs,None) # 预测的点
            prompt_labels = labels_choose.squeeze(0).to(device) #torch.Size([239, 1])
            nearest_points = find_nearest_points(outputs1['pred_coords'].cpu(), points_choose) # 从GT中匹配最近的点
            nearest_points_cat = torch.cat([nearest_points[i] for i in range(len(nearest_points))]).to(device)
            cell_nums = cell_nums.to(device) #torch.Size([16])

            '''Train image encoder'''                    
            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            # dimension hint for your future use
            # vision_feats: list: length = 3
            # vision_feats[0]: torch.Size([65536, batch, 32])
            # vision_feats[1]: torch.Size([16384, batch, 64])
            # vision_feats[2]: torch.Size([4096, batch, 256])
            # vision_pos_embeds[0]: torch.Size([65536, batch, 256])
            # vision_pos_embeds[1]: torch.Size([16384, batch, 256])
            # vision_pos_embeds[2]: torch.Size([4096, batch, 256])
            
            

            '''Train memory attention to condition on meomory bank'''         
            B = vision_feats[-1].size(1)  # batch size 
            
            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                
            else:
                for element in memory_bank_list:
                    to_cat_memory.append((element[0]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_features
                    to_cat_memory_pos.append((element[1]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_pos_enc
                    to_cat_image_embed.append((element[3]).cuda(non_blocking=True)) # image_embed

                memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)
 
                #vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, 64, 64) 
                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
                vision_feats_temp = vision_feats_temp.reshape(B, -1)

                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                
                similarity_scores = F.softmax(similarity_scores, dim=1) 
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  # Shape [batch_size, 16]

                memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))


                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                    )


            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            
            image_embed = feats[-1]
            high_res_feats = feats[:-1]
            
            # feats[0]: torch.Size([batch, 32, 256, 256]) #high_res_feats part1
            # feats[1]: torch.Size([batch, 64, 128, 128]) #high_res_feats part2
            # feats[2]: torch.Size([batch, 256, 64, 64]) #image_embed


            '''prompt encoder'''         
            with torch.no_grad():
                #points=(coords_torch, labels_torch)
                points=(nearest_points_cat.permute(1,0,2), prompt_labels.permute(1,0)) # input shape: ((batch, n, 2), (batch, n))
                flag = True

                se, de = net.sam_prompt_encoder(
                    points=points, #(coords_torch, labels_torch)
                    boxes=None,
                    masks=None,
                    batch_size=B,
                )
            # dimension hint for your future use
            # se: torch.Size([batch, n+1, 256])
            # de: torch.Size([batch, 256, 64, 64])



            
            '''train mask decoder'''       
            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False, # args.multimask_output if you want multiple masks
                    repeat_image=False,  # the image is already batched
                    cell_nums=cell_nums,
                    high_res_features = high_res_feats
                )
            # dimension hint for your future use
            # low_res_multimasks: torch.Size([batch, multimask_output, 256, 256])
            # iou_predictions.shape:torch.Size([batch, multimask_output])
            # sam_output_tokens.shape:torch.Size([batch, multimask_output, 256])
            # object_score_logits.shape:torch.Size([batch, 1])
            
            
            # resize prediction
            pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            

            '''memory encoder'''       
            # new caluculated memory features
            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks,
                is_mask_from_pts=flag)  
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]
                
            #maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features
            maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
            #maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
            maskmem_pos_enc = maskmem_pos_enc[0]
            maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)


            # add single maskmem_features, maskmem_pos_enc, iou
            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                             (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                             iou_predictions[batch, 0],
                                             image_embed[batch].reshape(-1).detach()])
            
            else:
                for batch in range(maskmem_features.size(0)):
                    
                    # current simlarity matrix in existing memory bank
                    memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                    memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                    # normalise
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                         memory_bank_maskmem_features_norm.t())

                    # replace diagonal (diagnoal always simiarity = 1)
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                    # first find the minimum similarity from memory feature and the maximum similarity from memory bank
                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores) 
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    # replace with less similar object
                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        # soft iou, not stricly greater than current iou
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index) 
                            memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                                     (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                                     iou_predictions[batch, 0],
                                                     image_embed[batch].reshape(-1).detach()])

            # backpropagation
            loss_dict = criterion(outputs1, targets,pred,masks, epoch)
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_reduced = reduce_dict(loss_dict)

            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_value = losses_reduced.item()
            for k, v in loss_dict_reduced.items():
                log_info[k] = log_info.get(k, 0) + v.item()

            optimizer.zero_grad()
            losses.backward()
            if args.clip_grad > 0:  # clip gradient
                torch.nn.utils.clip_grad_norm_(point_net.parameters(), args.clip_grad)
            optimizer.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            pbar.update()

    return epoch_loss/len(train_loader)

def eval_point(args, point_net, net,matcher, train_loader, criterion,optimizer, epoch, writer):
    
    # train mode
    net.train()
    point_net.eval()
    #optimizer.zero_grad()
    criterion.eval()

    log_info = dict()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # init
    epoch_loss = 0
    memory_bank_list = []
    lossfunc = criterion_G
    #feat_sizes = [(256, 256), (128, 128), (64, 64)]
    feat_sizes = [(64, 64), (32, 32), (16, 16)]

    total_eiou = 0
    total_dice = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for data_iter_step, (images, inst_masks,points_choose,labels_choose, points_list, labels_list , cell_nums , masks, ori_shape) in enumerate(train_loader):
            
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            imgs = images.to(device) # cpm:torch.Size([8, 3, 512, 512])   pannuke:torch.Size([16, 3, 256, 256])
            masks = masks.unsqueeze(1).to(device)
            
            outputs1,feats_origin,image_embedding,feats = point_net(imgs,None) # 预测的点
            prompt_labels = labels_choose.squeeze(0).to(device) #torch.Size([239, 1])
            nearest_points = find_nearest_points(outputs1['pred_coords'].cpu(), points_choose) # 从GT中匹配最近的点
            nearest_points_cat = torch.cat([nearest_points[i] for i in range(len(nearest_points))]).to(device)
            cell_nums = cell_nums.to(device) #torch.Size([16])

            cell_nums = cell_nums.to(device) #torch.Size([16])

            '''Train image encoder'''                    
            backbone_out = net.forward_image(imgs)
            _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
            # dimension hint for your future use
            # vision_feats: list: length = 3
            # vision_feats[0]: torch.Size([65536, batch, 32])
            # vision_feats[1]: torch.Size([16384, batch, 64])
            # vision_feats[2]: torch.Size([4096, batch, 256])
            # vision_pos_embeds[0]: torch.Size([65536, batch, 256])
            # vision_pos_embeds[1]: torch.Size([16384, batch, 256])
            # vision_pos_embeds[2]: torch.Size([4096, batch, 256])
            
            

            '''Train memory attention to condition on meomory bank'''         
            B = vision_feats[-1].size(1)  # batch size 
            
            if len(memory_bank_list) == 0:
                vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                
            else:
                for element in memory_bank_list:
                    to_cat_memory.append((element[0]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_features
                    to_cat_memory_pos.append((element[1]).cuda(non_blocking=True).flatten(2).permute(2, 0, 1)) # maskmem_pos_enc
                    to_cat_image_embed.append((element[3]).cuda(non_blocking=True)) # image_embed

                memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)
 
                #vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, 64, 64) 
                vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
                vision_feats_temp = vision_feats_temp.reshape(B, -1)

                image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()
                
                similarity_scores = F.softmax(similarity_scores, dim=1) 
                sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  # Shape [batch_size, 16]

                memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))


                vision_feats[-1] = net.memory_attention(
                    curr=[vision_feats[-1]],
                    curr_pos=[vision_pos_embeds[-1]],
                    memory=memory,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=0
                    )


            feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                     for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
            
            image_embed = feats[-1]
            high_res_feats = feats[:-1]
            
            # feats[0]: torch.Size([batch, 32, 256, 256]) #high_res_feats part1
            # feats[1]: torch.Size([batch, 64, 128, 128]) #high_res_feats part2
            # feats[2]: torch.Size([batch, 256, 64, 64]) #image_embed


            '''prompt encoder'''         
            with torch.no_grad():
                #points=(coords_torch, labels_torch)
                points=(nearest_points_cat.permute(1,0,2), prompt_labels.permute(1,0)) # input shape: ((batch, n, 2), (batch, n))
                flag = True

                se, de = net.sam_prompt_encoder(
                    points=points, #(coords_torch, labels_torch)
                    boxes=None,
                    masks=None,
                    batch_size=B,
                )
            # dimension hint for your future use
            # se: torch.Size([batch, n+1, 256])
            # de: torch.Size([batch, 256, 64, 64])



            
            '''train mask decoder'''       
            low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False, # args.multimask_output if you want multiple masks
                    repeat_image=False,  # the image is already batched
                    cell_nums=cell_nums,
                    high_res_features = high_res_feats
                )
            # dimension hint for your future use
            # low_res_multimasks: torch.Size([batch, multimask_output, 256, 256])
            # iou_predictions.shape:torch.Size([batch, multimask_output])
            # sam_output_tokens.shape:torch.Size([batch, multimask_output, 256])
            # object_score_logits.shape:torch.Size([batch, 1])
            
            
            # resize prediction
            pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
            high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            

            '''memory encoder'''       
            # new caluculated memory features
            maskmem_features, maskmem_pos_enc = net._encode_new_memory(
                current_vision_feats=vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_multimasks,
                is_mask_from_pts=flag)  
            # dimension hint for your future use
            # maskmem_features: torch.Size([batch, 64, 64, 64])
            # maskmem_pos_enc: [torch.Size([batch, 64, 64, 64])]
                
            #maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features
            maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
            #maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
            maskmem_pos_enc = maskmem_pos_enc[0]
            maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)


            # add single maskmem_features, maskmem_pos_enc, iou
            if len(memory_bank_list) < args.memory_bank_size:
                for batch in range(maskmem_features.size(0)):
                    memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                             (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                             iou_predictions[batch, 0],
                                             image_embed[batch].reshape(-1).detach()])
            
            else:
                for batch in range(maskmem_features.size(0)):
                    
                    # current simlarity matrix in existing memory bank
                    memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                    memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                    # normalise
                    memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                    current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                         memory_bank_maskmem_features_norm.t())

                    # replace diagonal (diagnoal always simiarity = 1)
                    current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                    diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                    current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                    # first find the minimum similarity from memory feature and the maximum similarity from memory bank
                    single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                    similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                    min_similarity_index = torch.argmin(similarity_scores) 
                    max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                    # replace with less similar object
                    if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                        # soft iou, not stricly greater than current iou
                        if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index) 
                            memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                                     (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                                     iou_predictions[batch, 0],
                                                     image_embed[batch].reshape(-1).detach()])
            #outputs1,feats_origin,image_embedding,feats = point_net(imgs)
            '''Train image encoder'''                    

            pred = (pred> 0.5).float()
            temp = eval_seg(pred, masks, threshold)
            total_eiou += temp[0]
            total_dice += temp[1]

            '''vis images'''
            namecat = 'Test'
            img_name = str(data_iter_step)
            namecat = namecat + img_name + '+'
            vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=None)
                            
            pbar.update()

    return  tuple([total_eiou/len(train_loader) , total_dice/len(train_loader)])




def validation_sam(args, cfgs, val_loader, epoch, point_net, net: nn.Module,num_classes,iou_threshold, calc_map=True, clean_dir=True):
    # eval mode
    point_net.eval()
    net.eval()

    n_val = len(val_loader) 
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    GPUdevice = torch.device('cuda:' + str(args.gpu_device))

    # init
    lossfunc = criterion_G
    memory_bank_list = []
    #feat_sizes = [(256, 256), (128, 128), (64, 64)]
    feat_sizes = [(64, 64), (32, 32), (16, 16)]
    total_loss = 0
    total_eiou = 0
    total_dice = 0


    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, (images,img_seg,inst_maps, type_maps,gt_points, labels, bi_masks, ori_shape,file_inds,name) in enumerate(val_loader):
        #for ind, pack in enumerate(val_loader):
            to_cat_memory = []
            to_cat_memory_pos = []
            to_cat_image_embed = []

            '''
            name = pack['image_meta_dict']['filename_or_obj']
            imgs = pack['image'].to(dtype = torch.float32, device = GPUdevice)
            masks = pack['mask'].to(dtype = torch.float32, device = GPUdevice)
            if 'pt' in pack:
                pt_temp = pack['pt'].to(device = GPUdevice)
                pt = pt_temp.unsqueeze(1)
                point_labels_temp = pack['p_label'].to(device = GPUdevice)
                point_labels = point_labels_temp.unsqueeze(1)
                coords_torch = torch.as_tensor(pt, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            else:
                coords_torch = None
                labels_torch = None
            '''
            imgs = img_seg.to(device)
            masks = bi_masks.to(device).unsqueeze(0).float()
            

            '''test'''
            with torch.no_grad():

                """ image encoder """
                backbone_out = net.forward_image(imgs)
                _, vision_feats, vision_pos_embeds, _ = net._prepare_backbone_features(backbone_out)
                B = vision_feats[-1].size(1) 

                """ memory condition """
                if len(memory_bank_list) == 0:
                    vision_feats[-1] = vision_feats[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")
                    vision_pos_embeds[-1] = vision_pos_embeds[-1] + torch.nn.Parameter(torch.zeros(1, B, net.hidden_dim)).to(device="cuda")

                else:
                    for element in memory_bank_list:
                        maskmem_features = element[0]
                        maskmem_pos_enc = element[1]
                        to_cat_memory.append(maskmem_features.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_memory_pos.append(maskmem_pos_enc.cuda(non_blocking=True).flatten(2).permute(2, 0, 1))
                        to_cat_image_embed.append((element[3]).cuda(non_blocking=True)) # image_embed
                        
                    memory_stack_ori = torch.stack(to_cat_memory, dim=0)
                    memory_pos_stack_ori = torch.stack(to_cat_memory_pos, dim=0)
                    image_embed_stack_ori = torch.stack(to_cat_image_embed, dim=0)

                    #vision_feats_temp = vision_feats[-1].permute(1, 0, 2).view(B, -1, 64, 64)
                    vision_feats_temp = vision_feats[-1].permute(1, 0, 2).reshape(B, -1, 64, 64)
                    vision_feats_temp = vision_feats_temp.reshape(B, -1)

                    image_embed_stack_ori = F.normalize(image_embed_stack_ori, p=2, dim=1)
                    vision_feats_temp = F.normalize(vision_feats_temp, p=2, dim=1)
                    similarity_scores = torch.mm(image_embed_stack_ori, vision_feats_temp.t()).t()

                    similarity_scores = F.softmax(similarity_scores, dim=1) 
                    sampled_indices = torch.multinomial(similarity_scores, num_samples=B, replacement=True).squeeze(1)  # Shape [batch_size, 16]

                    memory_stack_ori_new = (memory_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory = memory_stack_ori_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))

                    memory_pos_stack_new = (memory_pos_stack_ori[sampled_indices].squeeze(3).permute(1, 2, 0, 3))
                    memory_pos = memory_pos_stack_new.reshape(-1, memory_stack_ori_new.size(2), memory_stack_ori_new.size(3))



                    vision_feats[-1] = net.memory_attention(
                        curr=[vision_feats[-1]],
                        curr_pos=[vision_pos_embeds[-1]],
                        memory=memory,
                        memory_pos=memory_pos,
                        num_obj_ptr_tokens=0
                        )

                feats = [feat.permute(1, 2, 0).view(B, -1, *feat_size) 
                        for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]
                
                image_embed = feats[-1]
                high_res_feats = feats[:-1]

                """ prompt encoder """
                flag = True
                #points = (coords_torch, labels_torch)
                points = (gt_points.to(device), labels.to(device))

                se, de = net.sam_prompt_encoder(
                    points=points, 
                    boxes=None,
                    masks=None,
                    batch_size=B,
                )

                low_res_multimasks, iou_predictions, sam_output_tokens, object_score_logits = net.sam_mask_decoder(
                    image_embeddings=image_embed,
                    image_pe=net.sam_prompt_encoder.get_dense_pe(), 
                    sparse_prompt_embeddings=se,
                    dense_prompt_embeddings=de, 
                    multimask_output=False, 
                    repeat_image=False,  
                    cell_nums=torch.as_tensor([len(gt_points)]).to(gt_points.device),
                    high_res_features = high_res_feats
                )

                # prediction
                pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size))
                high_res_multimasks = F.interpolate(low_res_multimasks, size=(args.image_size, args.image_size),
                                                mode="bilinear", align_corners=False)
            
                """ memory encoder """
                maskmem_features, maskmem_pos_enc = net._encode_new_memory( 
                    current_vision_feats=vision_feats,
                    feat_sizes=feat_sizes,
                    pred_masks_high_res=high_res_multimasks,
                    is_mask_from_pts=flag)  
                    
                #maskmem_features = maskmem_features.to(torch.bfloat16)
                maskmem_features = maskmem_features
                maskmem_features = maskmem_features.to(device=GPUdevice, non_blocking=True)
                #maskmem_pos_enc = maskmem_pos_enc[0].to(torch.bfloat16)
                maskmem_pos_enc = maskmem_pos_enc[0]
                maskmem_pos_enc = maskmem_pos_enc.to(device=GPUdevice, non_blocking=True)


                """ memory bank """
                if len(memory_bank_list) < 16:
                    for batch in range(maskmem_features.size(0)):
                        memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                 (maskmem_pos_enc[batch].unsqueeze(0)),
                                                 iou_predictions[batch, 0],
                                                 image_embed[batch].reshape(-1).detach()])
                
                else:
                    for batch in range(maskmem_features.size(0)):
                        
                        memory_bank_maskmem_features_flatten = [element[0].reshape(-1) for element in memory_bank_list]
                        memory_bank_maskmem_features_flatten = torch.stack(memory_bank_maskmem_features_flatten)

                        memory_bank_maskmem_features_norm = F.normalize(memory_bank_maskmem_features_flatten, p=2, dim=1)
                        current_similarity_matrix = torch.mm(memory_bank_maskmem_features_norm,
                                                             memory_bank_maskmem_features_norm.t())

                        current_similarity_matrix_no_diag = current_similarity_matrix.clone()
                        diag_indices = torch.arange(current_similarity_matrix_no_diag.size(0))
                        current_similarity_matrix_no_diag[diag_indices, diag_indices] = float('-inf')

                        single_key_norm = F.normalize(maskmem_features[batch].reshape(-1), p=2, dim=0).unsqueeze(1)
                        similarity_scores = torch.mm(memory_bank_maskmem_features_norm, single_key_norm).squeeze()
                        min_similarity_index = torch.argmin(similarity_scores) 
                        max_similarity_index = torch.argmax(current_similarity_matrix_no_diag[min_similarity_index])

                        if similarity_scores[min_similarity_index] < current_similarity_matrix_no_diag[min_similarity_index][max_similarity_index]:
                            if iou_predictions[batch, 0] > memory_bank_list[max_similarity_index][2] - 0.1:
                                memory_bank_list.pop(max_similarity_index) 
                                memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                         (maskmem_pos_enc[batch].unsqueeze(0)),
                                                         iou_predictions[batch, 0],
                                                         image_embed[batch].reshape(-1).detach()])

                # binary mask and calculate loss, iou, dice
                total_loss += lossfunc(pred, masks)
                pred = (pred> 0.5).float()
                temp = eval_seg(pred, masks, threshold)
                total_eiou += temp[0]
                total_dice += temp[1]

                '''vis images'''
                if ind % args.vis == 0:
                    namecat = 'Test'
                    for na in name:
                        img_name = na
                        namecat = namecat + img_name + '+'
                    vis_image(imgs,pred, masks, os.path.join(args.path_helper['sample_path'], namecat+'epoch+' +str(epoch) + '.jpg'), reverse=False, points=None)
                            
            pbar.update()

    return total_loss/ n_val , tuple([total_eiou/n_val, total_dice/n_val])

@staticmethod
def find_nearest_points(pred_coords, points_choose):
    batch_size = pred_coords.shape[0]
    nearest_points = []

    for i in range(batch_size):
        pred_points = pred_coords[i].float()  # shape: (1024, 2)
        chosen_points = points_choose[i]  # shape: (25, 1, 2)

        chosen_points = chosen_points.view(-1, 2).float()  # shape: (25, 2)

        # Calculate pairwise distances
        distances = torch.cdist(pred_points.unsqueeze(0), chosen_points.unsqueeze(0)).squeeze(0)  # shape: (1024, 25)

        # Find the indices of the nearest points in pred_
        nearest_indices = torch.argmin(distances, dim=0)  # shape: (25,)

        # Gather the nearest points
        nearest_points_batch = pred_points[nearest_indices].unsqueeze(1)  # shape: (25, 1, 2)
        
        nearest_points.append(nearest_points_batch)

    # nearest_points = torch.stack(nearest_points)  # shape: (batch_size, 25, 1, 2)
    return nearest_points
