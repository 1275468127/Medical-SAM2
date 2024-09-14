
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


def train_sam(args, point_net,net: nn.Module, matcher, train_loader, criterion,optimizer, epoch, writer, model_ema=None):
    # train mode
    point_net.train()
    net.train()
    criterion.train()
    optimizer.zero_grad()

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
            metric_logger.log_every(train_loader, args.print_freq, header)):
            
            to_cat_memory = []
            to_cat_memory_pos = []

            # input image and gt masks
            images = images.to(device) # cpm:torch.Size([8, 3, 512, 512])   pannuke:torch.Size([16, 3, 256, 256])
            masks = masks.to(device)
            targets = {
                'gt_masks': masks,
                'gt_nums': [len(points) for points in points_list],
                'gt_points': [points.view(-1, 2).to(device).float() for points in points_list],
                'gt_labels': [labels.to(device).long() for labels in labels_list],
            }
            inst_masks = inst_masks.to(device) #torch.Size([256, 256])

            # click prompt: unsqueeze to indicate only one click, add more click across this dimension
            outputs1,feats_origin,image_embedding,feats = point_net(images,None) # 预测的点

            prompt_labels = labels_choose.squeeze(0).to(device) #torch.Size([239, 1])
            nearest_points = find_nearest_points(outputs1['pred_coords'].cpu(), points_choose) # 从GT中匹配最近的点
            nearest_points_cat = torch.cat([nearest_points[i] for i in range(len(nearest_points))]).to(device)
            cell_nums = cell_nums.to(device) #torch.Size([16])


            '''Train image encoder'''                    
            backbone_out = net.forward_image(images)
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
                memory = torch.cat(to_cat_memory, dim=0)
                memory_pos = torch.cat(to_cat_memory_pos, dim=0)

                
                memory = memory.repeat(1, B, 1) 
                memory_pos = memory_pos.repeat(1, B, 1) 


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
                points=(nearest_points_cat, prompt_labels) # input shape: ((batch, n, 2), (batch, n))
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
            values, indices = torch.max(iou_predictions, dim=1)
            iou_predictions = values.mean()
            
            # resize prediction
            pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size),mode="bilinear",align_corners=False)[:, 0]
            inst_pred = combine_mask(ori_shape,nearest_points_cat,pred,values)
            high_res_multimasks = (inst_pred.copy() > 0).astype(float)
            high_res_multimasks = torch.from_numpy(high_res_multimasks).unsqueeze(0).unsqueeze(0).to(device)

            vis_image(images,high_res_multimasks, masks.unsqueeze(0), os.path.join('/data/hhb/project/PS-SAM2/visi/train_small1.jpg'), reverse=False, points=None)

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
                                             iou_predictions])
            
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
                        if iou_predictions > memory_bank_list[max_similarity_index][2] - 0.1:
                            memory_bank_list.pop(max_similarity_index) 
                            memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)).detach(),
                                                     (maskmem_pos_enc[batch].unsqueeze(0)).detach(),
                                                     iou_predictions])

            # backpropagation
            loss_dict = criterion(outputs1, targets,pred,values,inst_masks.squeeze(0).to(torch.float32), epoch)
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

    return {k: v / len(train_loader) for k, v in log_info.items()}




def validation_sam(args, cfgs, val_loader, epoch, point_net, net: nn.Module,num_classes,iou_threshold, calc_map=True, clean_dir=True):
    # eval mode
    point_net.eval()
    net.eval()

    for name, param in point_net.named_parameters():
        # print(name, param.mean().item())  # 检查参数是否为零或异常小
        print(name, param)#为什么打印的是mean啊

    cls_predictions = []
    cls_annotations = []

    cls_pn, cls_tn = list(torch.zeros(num_classes).to(device) for _ in range(2))
    cls_rn = torch.zeros(num_classes).to(device)

    det_pn, det_tn = list(torch.zeros(1).to(device) for _ in range(2))
    det_rn = torch.zeros(1).to(device)

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

    iou_scores = []

    binary_pq_scores = []  # image_id
    binary_dq_scores = []
    binary_sq_scores = []
    binary_aji_scores = []
    binary_aji_plus_scores = []
    binary_dice_scores = []

    aji_scores = []
    dice_scores = []
    aji_plus_scores = []
    excel_info = []
    save_frame_data = [] #保存表格用的

    n_val = len(val_loader) 
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for ind, (images,img_seg,inst_maps, type_maps,gt_points, labels, bi_masks, ori_shape,file_inds,name) in enumerate(val_loader):
            assert len(img_seg) == 1, 'batch size must be 1'

            to_cat_memory = []
            to_cat_memory_pos = []

            images = images.to(device)
            images_seg =img_seg.to(device)
            inst_maps = inst_maps
            type_maps = type_maps.squeeze(0)

            inst_maps = inst_maps.numpy()
            type_maps = F.one_hot(
                type_maps.type(torch.int64),
                num_classes + 1
            ).numpy()

            instance_types_nuclei = type_maps * np.expand_dims(inst_maps, -1)
            instance_types_nuclei = instance_types_nuclei.transpose(0, 3, 1, 2)  # b h w c

            box_feature_map = {}

            all_masks = []
            all_boxes = []
            all_scores = []
            all_classes = []
            all_inds = []

            #单张大图 语义的结果列表
            all_semantic_masks=[]
            all_semantic_boxes=[]
            all_semantic_scores=[]
            all_semantic_classes=[]
            all_semantic_inds=[]

            #单张大图 point的结果列表
            all_points = []
            all_points_scores = []
            all_points_class = []
            all_ori_points = []
            all_ori_scores = []
            point_num = []

            #单张大图 用来后处理的id列表
            inds = torch.tensor([]).long() 
            processed_boxes = []
            point_id_map = {}  # 用于记录已处理的点及其ID
            next_id = 0  # 下一个可用的ID

            #对大图crop生成小图
            crop_boxes = crop_with_overlap(
                images_seg[0],
                args.crop_size,
                args.crop_size,
                args.overlap,
            ).tolist()

            for idx, crop_box in enumerate(crop_boxes):
                x1, y1, x2, y2 = crop_box
                crop_box_tuple = tuple(crop_box)
                box_feature_map[crop_box_tuple] = None
                #img = F.interpolate(images_seg[..., y1:y2, x1:x2].to(device), size=(1024,1024), mode='bilinear', align_corners=False)
                img = images_seg[..., y1:y2, x1:x2].to(device)

                ###model1预测点
                # 在256的小图中预测点
                with torch.no_grad():
                    pd_points, pd_scores, pd_classes, pd_masks,masks_ ,ori_points,ori_scores = predict(
                        point_net,
                        images_seg[..., y1:y2, x1:x2].to(device),
                        ori_shape=np.array((y2 - y1, x2 - x1)),
                        filtering=cfgs.test.filtering,
                        nms_thr=cfgs.test.nms_thr,
                        semantic_feature=box_feature_map[crop_box_tuple]

                    )

                #加上当前小图对应的偏移得到点在整张大图上的坐标。
                pd_points[:, 0] += x1
                pd_points[:, 1] += y1

                ori_points[:, 0] += x1
                ori_points[:, 1] += y1

                # 检查pd_points是否出现在之前的任何一个box里
                bool_mask = np.ones(len(pd_points), dtype=bool)
                for prev_box in processed_boxes:
                    px1, py1, px2, py2 = prev_box
                    # 如果pd_points在之前的box里，设置mask为False
                    bool_mask &= ~((pd_points[:, 0] >= px1+1) & (pd_points[:, 0] <= px2-1) &
                            (pd_points[:, 1] >= py1+1) & (pd_points[:, 1] <= py2-1))

                pd_points = pd_points[bool_mask]
                pd_scores = pd_scores[bool_mask]
                pd_classes = pd_classes[bool_mask]

                point_num.append(pd_points.shape[0])
                
                all_points.append(pd_points)
                all_points_scores.append(pd_scores)
                all_points_class.append(pd_classes)
                all_ori_points.append(ori_points)
                final_all_ori_points = np.vstack(all_ori_points)
                all_ori_scores.append(ori_scores)
                final_all_ori_scores = np.vstack(all_ori_scores)
                current_all_points = np.vstack(all_points)
                current_all_points_scores = np.concatenate(all_points_scores)
                current_all_points_class = np.concatenate(all_points_class)
                current_all_points, current_all_points_scores, current_all_points_class = point_nms(current_all_points, current_all_points_scores, current_all_points_class, cfgs.test.nms_thr)

                #记录最终选的点的ID，这个ID是相对于大图的ID，而不是每个小图都重新编号的。
                current_inds = []
                for point in current_all_points:
                    point_tuple = tuple(point)
                    if point_tuple not in point_id_map:
                        point_id_map[point_tuple] = next_id
                        next_id += 1
                    current_inds.append(point_id_map[point_tuple])

                # 将当前的ID列表转换为Tensor
                current_inds = torch.tensor(current_inds).long()

                inds = torch.cat((inds, current_inds)) # 记录所有点的ID

                # 将当前box添加到已处理的box列表中
                processed_boxes.append(crop_box)

                prompt_points = torch.from_numpy(current_all_points).unsqueeze(1)

                #筛选位于当前小图上的点（点加上偏移可能会出这个图）
                keep = (prompt_points[..., 0] >= x1) & (prompt_points[..., 0] < x2) & \
                        (prompt_points[..., 1] >= y1) & (prompt_points[..., 1] < y2)
                keep = keep.squeeze(1)
    
                if keep.sum() == 0 or keep.sum()==1: #这个东西总在报错
                    continue

                sub_prompt_points = (prompt_points[keep] - torch.as_tensor([x1, y1])).to(device) #在筛属于当前crop的点
                prompt_labels = torch.zeros(sub_prompt_points.size(0), dtype=torch.int).to(device)
                sub_prompt_labels = prompt_labels.unsqueeze(1)

                '''test'''
                with torch.no_grad():

                    """ image encoder """
                    backbone_out = net.forward_image(img)
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
                        memory = torch.cat(to_cat_memory, dim=0)
                        memory_pos = torch.cat(to_cat_memory_pos, dim=0)

                        memory = memory.repeat(1, B, 1) 
                        memory_pos = memory_pos.repeat(1, B, 1) 

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
                    points = (sub_prompt_points, sub_prompt_labels)

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
                        cell_nums=torch.as_tensor([len(sub_prompt_points)]).to(sub_prompt_points.device),
                        high_res_features = high_res_feats
                    )
                    values, indices = torch.max(iou_predictions, dim=1)
                    iou_predictions = values.mean()

                    # prediction
                    pred = F.interpolate(low_res_multimasks,size=(args.out_size,args.out_size),mode="bilinear",align_corners=False)[:, 0]
                    inst_pred = combine_mask(ori_shape,sub_prompt_points,pred,values)
                    # prediction
                    high_res_multimasks = (inst_pred.copy() > 0).astype(float)

                    high_res_multimasks = torch.from_numpy(high_res_multimasks).unsqueeze(0).unsqueeze(0).to(device)
                
                    GT1 = F.interpolate(torch.tensor(inst_maps).unsqueeze(0)[..., y1:y2, x1:x2].to(device), size=(args.out_size, args.out_size),
                                                        mode="bilinear", align_corners=False)
                    GT1 = GT1 >= 1
                    
                    #visualize_points_on_images(GT1, sub_prompt_points.permute(1, 0, 2), os.path.join('/data/hhb/image_point.jpg'))
                    vis_image(img,high_res_multimasks, GT1, os.path.join('/data/hhb/epoch+' +str(epoch) + '_small.jpg'), reverse=False, points=None)

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
                                                    iou_predictions])
                    
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
                                if iou_predictions > memory_bank_list[max_similarity_index][2] - 0.1:
                                    memory_bank_list.pop(max_similarity_index) 
                                    memory_bank_list.append([(maskmem_features[batch].unsqueeze(0)),
                                                            (maskmem_pos_enc[batch].unsqueeze(0)),
                                                            iou_predictions])
                    # 先扩成大图（其他区域用背景填充），然后存储结果后续合并。  
                    masks = mask_post_eval(current_all_points_class[keep],current_inds[keep],crop_box,ori_shape,sub_prompt_points,pred,values)
                    for mask_data in masks:
                        all_scores.append(mask_data['predicted_iou'])
                        all_masks.append(mask_data['segmentation'][:ori_shape[0, 0], :ori_shape[0, 1]])
                        all_boxes.append(mask_data['bbox'])
                        all_classes.append(mask_data['categories'])
                        all_inds.append(mask_data['inds'])

            print(point_num)
            #合并成大图
            all_boxes = torch.as_tensor(all_boxes)
            all_scores = torch.as_tensor(all_scores)
            all_inds = np.asarray(all_inds)
            unique_inds, counts = np.unique(all_inds, return_counts=True)

            # all_semantic_inds = np.asarray(all_semantic_inds)
            # unique_semantic_inds, semantic_counts = np.unique(all_semantic_inds, return_counts=True)

            # first-aspect NMS 取分数最大的实例，就是合并基于每个点提示的分割图
            keep_prior = np.ones(len(all_inds), dtype=bool)
            for i in np.where(counts > 1)[0]:
                inds = np.where(all_inds == unique_inds[i])[0]
                inds = np.delete(inds, np.argmax(all_scores[inds]))
                keep_prior[inds] = False
            keep_prior = torch.from_numpy(keep_prior)

            # keep_semantic_prior = np.ones(len(all_semantic_inds), dtype=bool)
            # for i in np.where(semantic_counts > 1)[0]:
            #     inds = np.where(all_semantic_inds == unique_semantic_inds[i])[0]
            #     inds = np.delete(inds, np.argmax(all_semantic_scores[inds]))
            #     keep_semantic_prior[inds] = False
            # keep_semantic_prior = torch.from_numpy(keep_semantic_prior)

            all_boxes = all_boxes[keep_prior]
            all_scores = all_scores[keep_prior]
            all_masks = [all_masks[ind] for ind in np.where(keep_prior)[0]]

            # all_semantic_boxes = all_semantic_boxes[keep_semantic_prior]
            # all_semantic_scores = all_semantic_scores[keep_semantic_prior]
            # all_semantic_masks = [all_semantic_masks[ind] for ind in np.where(keep_semantic_prior)[0]]


            # second-aspect NMS 合并不同的box，就是拼成一个1024的图
            if len(all_boxes.shape) == 1:
                # all_boxes 是一维的，使用一维索引
                cross_categories = torch.zeros_like(all_boxes)
            else:
                # all_boxes 是多维的，使用多维索引
                cross_categories = torch.zeros_like(all_boxes[:, 0])
            keep_by_nms = batched_nms(
                all_boxes.float(),
                all_scores,
                cross_categories,  # apply cross categories
                iou_threshold=iou_threshold
            ).numpy()
            order = keep_by_nms[::-1]
            b_inst_map = np.zeros_like(inst_maps[0], dtype=int)
            for iid, ind in enumerate(order):
                if b_inst_map[all_masks[ind]].all() == 0:
                    b_inst_map[all_masks[ind]] = iid + 1

            # binary mask and calculate loss, iou, dice
            if len(np.unique(inst_maps[0])) == 1 or len(b_inst_map)==1 or len(b_inst_map)==0 or len(np.unique(inst_maps[0])) == 0 or len(np.unique(b_inst_map))==1:
                bpq_tmp = np.nan
                bdq_tmp = np.nan
                bsq_tmp = np.nan 
                baji_tmp = np.nan 
                baji_plus_tmp  = np.nan 
                bdice_tmp  = np.nan 
            else:
                [bdq_tmp, bsq_tmp, bpq_tmp], _ = get_fast_pq(
                    remap_label(inst_maps[0]),
                    remap_label(b_inst_map)
                )
                bdice_tmp = get_dice_2(
                    remap_label(inst_maps[0]),
                    remap_label(b_inst_map)
                )
                baji_plus_tmp = get_fast_aji_plus(
                    remap_label(inst_maps[0]),
                    remap_label(b_inst_map)
                )
                # print(np.unique(inst_maps[0]))
                # print(np.unique(b_inst_map))
                baji_tmp = get_fast_aji(
                    remap_label(inst_maps[0]),
                    remap_label(b_inst_map)
                )

                binary_dq_scores.append(bdq_tmp)
                binary_sq_scores.append(bsq_tmp)

                binary_pq_scores.append(bpq_tmp)
                binary_aji_plus_scores.append(baji_plus_tmp)
                binary_dice_scores.append(bdice_tmp)
                binary_aji_scores.append(baji_tmp)
                excel_info.append(
                    (name,
                    bpq_tmp)
                )

            #point 指标
            all_points = np.vstack(all_points)
            all_points_scores = np.concatenate(all_points_scores)
            all_points_class = np.concatenate(all_points_class)
            pd_points = all_points
            prompt_points = torch.from_numpy(pd_points).unsqueeze(1)
            pd_scores = all_points_scores
            pd_classes = all_points_class

            gt_points = gt_points[0].reshape(-1, 2).numpy()
            labels = labels[0].numpy()

            cls_annotations.append({'points': gt_points, 'labels': labels})

            cls_pred_sample = []
            for c in range(cfgs.data.num_classes):
                ind = (pd_classes == c)
                category_pd_points = pd_points[ind]
                category_pd_scores = pd_scores[ind]
                category_gt_points = gt_points[labels == c]

                cls_pred_sample.append(np.concatenate([category_pd_points, category_pd_scores[:, None]], axis=-1))

                pred_num, gd_num = len(category_pd_points), len(category_gt_points)
                cls_pn[c] += pred_num
                cls_tn[c] += gd_num

                if pred_num and gd_num:
                    cls_right_nums,_ = get_tp(category_pd_points, category_pd_scores, category_gt_points, thr=cfgs.test.match_dis)
                    cls_rn[c] += torch.tensor(cls_right_nums, device=cls_rn.device)

            cls_predictions.append(cls_pred_sample)

            det_pn += len(pd_points)
            det_tn += len(gt_points)

            unmatched_points = np.array([]) 
            if len(pd_points) and len(gt_points):
                det_right_nums, unmatched_points = get_tp(pd_points, pd_scores, gt_points, thr=cfgs.test.match_dis)
                det_rn += torch.tensor(det_right_nums, device=det_rn.device)

            csv_filename = 'cpm_per_img_score.csv'
            df = pd.DataFrame(save_frame_data, columns=["image","bdice_tmp", "baji_tmp", "bdq_tmp", "bsq_tmp", "bpq_tmp", "baji_plus_tmp"])
            df.to_csv(csv_filename, index=False)

            print("*"*10+"binary指标："+"*"*10)
            seg_dice = np.nanmean(binary_dice_scores)
            seg_aji = np.nanmean(binary_aji_scores)
            seg_dq = np.nanmean(binary_dq_scores)
            seg_sq = np.nanmean(binary_sq_scores)
            seg_pq = np.nanmean(binary_pq_scores)
            seg_aji_p = np.nanmean(binary_aji_plus_scores)

            print("dice:",f"{seg_dice*100:.2f}" ,end=" ")
            print("aji:",f"{seg_aji*100:.2f}" ,end=" ")
            print("aji_p:",f"{seg_aji_p*100:.2f}" ,end=" ")
            print("dq:",f"{seg_dq*100:.2f}" ,end=" ")
            print("sq:",f"{seg_sq*100:.2f}" ,end=" ")
            print("pq:",f"{seg_pq*100:.2f}" )

            # 计算检测和分类的指标
            eps = 1e-7
            det_r = det_rn / (det_tn + eps)
            det_p = det_rn / (det_pn + eps)
            det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)

            det_r = det_r.cpu().numpy() * 100
            det_p = det_p.cpu().numpy() * 100
            det_f1 = det_f1.cpu().numpy() * 100

            cls_r = cls_rn / (cls_tn + eps)
            cls_p = cls_rn / (cls_pn + eps)
            cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

            cls_r = cls_r.cpu().numpy() * 100
            cls_p = cls_p.cpu().numpy() * 100
            cls_f1 = cls_f1.cpu().numpy() * 100

            table = pt.PrettyTable()
            table.add_column('CLASS', ["nuclei"])

            table.add_column('Precision', cls_p.round(2))
            table.add_column('Recall', cls_r.round(2))
            table.add_column('F1', cls_f1.round(2))

            table.add_row(['---'] * 4)

            det_p, det_r, det_f1 = det_p.round(2)[0], det_r.round(2)[0], det_f1.round(2)[0]
            cls_p, cls_r, cls_f1 = cls_p.mean().round(2), cls_r.mean().round(2), cls_f1.mean().round(2)

            table.add_row(['Det', det_p, det_r, det_f1])
            table.add_row(['Cls', cls_p, cls_r, cls_f1])
            print(table)

            if calc_map:
                mAP = eval_map(cls_predictions, cls_annotations, cfgs.test.match_dis)[0]
                print(f'mAP: {round(mAP * 100, 2)}')

            metrics = {'Det': [det_p, det_r, det_f1], 'Cls': [cls_p, cls_r, cls_f1],
                        'IoU': (np.mean(iou_scores) * 100).round(2) ,'SEG':[seg_dice,seg_aji,seg_aji_p,seg_dq,seg_sq,seg_pq]}
            
            '''vis images'''
            vis_image(img_seg.to(device),torch.tensor(b_inst_map).unsqueeze(0).unsqueeze(0).to(device), torch.tensor(inst_maps).unsqueeze(0).to(device), os.path.join('/data/hhb/epoch+' +str(epoch) + '_eval.jpg'), reverse=False, points=None)
            pbar.update()

    return metrics , table.get_string()

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

def mask_post_eval(cell_types,sub_inds,crop_box, ori_shape, points, pred, iou_predictions,mask_threshold: float = .0,stability_score_offset: float = 1.0,box_nms_thresh: float = 1.0,pred_iou_thresh: float = 0.0,stability_score_thresh: float = 0.0):
    orig_h, orig_w = ori_shape[0]
    # Serialize predictions and store in MaskData
    #cell_types = np.zeros(points.shape[0], dtype=np.int64)
    #sub_inds = torch.arange(points.shape[0], dtype=torch.int64)
    mask_data = MaskData(
        masks=pred,
        iou_preds=iou_predictions,
        points=points,
        categories=cell_types,
        inds=sub_inds
    )

    # Filter by predicted IoU
    if pred_iou_thresh > 0.0:
        keep_mask = mask_data["iou_preds"] > pred_iou_thresh
        mask_data.filter(keep_mask)

    # Calculate stability score
    mask_data["stability_score"] = calculate_stability_score(
        mask_data["masks"], mask_threshold, stability_score_offset
    )

    if stability_score_thresh > 0.0:
        keep_mask = mask_data["stability_score"] >= stability_score_thresh
        mask_data.filter(keep_mask)

    # Threshold masks and calculate boxes
    mask_data["masks"] = mask_data["masks"] > mask_threshold
    mask_data["boxes"] = batched_mask_to_box(mask_data["masks"])

    # Filter boxes that touch crop boundaries
    #keep_mask = ~is_box_near_crop_edge(batch_data["boxes"], crop_box, [0, 0, orig_w, orig_h], atol=7)

    #if bi_masks!=None and (not torch.all(keep_mask)) :
    #        batch_data.filter(keep_mask)

    # Compress to RLE
    mask_data["masks"] = uncrop_masks(mask_data["masks"], crop_box, orig_h, orig_w)
    mask_data["rles"] = mask_to_rle_pytorch(mask_data["masks"])
    del mask_data["masks"]


    keep_by_nms = batched_nms(
        mask_data["boxes"].float(),
        mask_data["iou_preds"],
        torch.zeros_like(mask_data["boxes"][:, 0]),  # apply cross categories
        iou_threshold=box_nms_thresh
    )
    mask_data.filter(keep_by_nms)

    # Return to the original image frame
    mask_data["boxes"] = uncrop_boxes_xyxy(mask_data["boxes"], crop_box)
    mask_data["points"] = uncrop_points(mask_data["points"], crop_box)
    mask_data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(mask_data["rles"]))])
    mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]



    # Write mask records
    masks = []
    for idx in range(len(mask_data["segmentations"])):
        ann = {
            "segmentation": mask_data["segmentations"][idx],
            "area": area_from_rle(mask_data["rles"][idx]),
            "bbox": mask_data["boxes"][idx].tolist(),
            "predicted_iou": mask_data["iou_preds"][idx].item(),
            "point_coords": [mask_data["points"][idx].tolist()],
            "stability_score": mask_data["stability_score"][idx].item(),
            "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            'categories': mask_data['categories'][idx].tolist(),
            'inds': mask_data['inds'][idx].tolist()
        }
        masks.append(ann)
    return masks

def combine_mask(ori_shape, points, pred, iou_predictions,mask_threshold: float = .0,stability_score_offset: float = 1.0,box_nms_thresh: float = 1.0):
    # Serialize predictions and store in MaskData
    cell_types = np.zeros(points.shape[0], dtype=np.int64)
    sub_inds = torch.arange(points.shape[0], dtype=torch.int64)
    mask_data = MaskData(
        masks=pred,
        iou_preds=iou_predictions,
        points=points,
        categories=cell_types,
        inds=sub_inds
    )
    # Calculate stability score
    mask_data["stability_score"] = calculate_stability_score(
        mask_data["masks"], mask_threshold, stability_score_offset
    )

    # Threshold masks and calculate boxes
    mask_data["masks"] = mask_data["masks"] > mask_threshold
    mask_data["boxes"] = batched_mask_to_box(mask_data["masks"])

    # Compress to RLE
    mask_data["rles"] = mask_to_rle_pytorch(mask_data["masks"])
    del mask_data["masks"]


    keep_by_nms = batched_nms(
        mask_data["boxes"].float(),
        mask_data["iou_preds"],
        torch.zeros_like(mask_data["boxes"][:, 0]),  # apply cross categories
        iou_threshold=box_nms_thresh
    )
    mask_data.filter(keep_by_nms)

    # Return to the original image frame
    mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]


    # Write mask records
    masks = []
    for idx in range(len(mask_data["segmentations"])):
        ann = {
            "segmentation": mask_data["segmentations"][idx],
            "area": area_from_rle(mask_data["rles"][idx]),
            "predicted_iou": mask_data["iou_preds"][idx].item(),
            "stability_score": mask_data["stability_score"][idx].item(),
            'categories': mask_data['categories'][idx].tolist(),
            'inds': mask_data['inds'][idx].tolist()
        }
        masks.append(ann)

    all_masks = []
    all_scores = []
    all_inds = []

    for mask_data in masks:
        all_scores.append(mask_data['predicted_iou'])
        all_masks.append(mask_data['segmentation'][:ori_shape[0, 0], :ori_shape[0, 1]])
        all_inds.append(mask_data['inds'])

    all_scores = torch.as_tensor(all_scores)
    all_inds = np.asarray(all_inds)
    unique_inds, counts = np.unique(all_inds, return_counts=True)

    keep_prior = np.ones(len(all_inds), dtype=bool)
    for i in np.where(counts > 1)[0]:
        inds = np.where(all_inds == unique_inds[i])[0]
        inds = np.delete(inds, np.argmax(all_scores[inds]))
        keep_prior[inds] = False
    keep_prior = torch.from_numpy(keep_prior)

    all_masks = [all_masks[ind] for ind in np.where(keep_prior)[0]]

    '''
    keep_by_nms = batched_nms(
        all_boxes.float(),
        all_scores,
        cross_categories,  # apply cross categories
        iou_threshold=iou_threshold
    ).numpy()
    order = keep_by_nms[::-1]
    '''
    pred_map = np.zeros((pred.shape[1], pred.shape[2]), dtype=int)

    for ind in np.where(keep_prior)[0]:
        if pred_map[all_masks[ind]].all() == 0:
            pred_map[all_masks[ind]] = ind + 1
            #pred_map[all_masks[ind]] = 1
    #pred_map = remap_label(pred_map)
    return pred_map

def crop_with_overlap(
        img,
        split_width,
        split_height,
        overlap
):
    def start_points(
            size,
            split_size,
            overlap
    ):
        points = [0]
        counter = 1
        stride = 256 - overlap
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                if split_size == size:
                    break
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

    _, img_h, img_w = img.shape

    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    crop_boxes = []
    for y in Y_points:
        for x in X_points:
            crop_boxes.append([x, y, min(x + split_width, img_w), min(y + split_height, img_h)])
    return np.asarray(crop_boxes)