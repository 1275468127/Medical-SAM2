import torch
import torch.nn as nn
from pytorch_toolbelt.losses import BinaryFocalLoss, DiceLoss

import torch.nn.functional as F
from sam2_train.modeling.matcher import build_matcher
from pytorch_toolbelt.losses import BinaryFocalLoss
from sam2_train.modeling.utils import is_dist_avail_and_initialized, get_world_size


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B]
        """

        p = torch.sigmoid(pred_mask[:, 0])
        intersection = torch.sum(p * ground_truth_mask, dim=(1, 2))
        union = torch.sum(p, dim=(1, 2)) + torch.sum(ground_truth_mask, dim=(1, 2)) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)

        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class Criterion(nn.Module):
    def __init__(self, num_classes, matcher, class_weight, loss_weight, reg_loss_type='l2'):
        super().__init__()
        self.matcher = matcher
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.reg_loss_type = reg_loss_type

        self.focal_loss = BinaryFocalLoss()
        self.dice_loss = DiceLoss('binary')
        self.iou_loss = MaskIoULoss()
        pos_weight = torch.ones([1]).cuda('cuda')*2
        self.lossfunc = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    def loss_reg(self, outputs, targets, indices, num_points):
        """ Regression loss """
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs['pred_coords'][idx]

        target_points = torch.cat([gt_points[J] for gt_points, (_, J) in zip(targets['gt_points'], indices)], dim=0)

        if self.reg_loss_type == 'l2':
            loss_pnt = F.mse_loss(src_points, target_points, reduction='none')
        else:
            loss_pnt = F.l1_loss(src_points, target_points, reduction='none')

        loss_dict =loss_pnt.sum() / (num_points + 1e-7)
        return loss_dict

    def loss_cls(self, outputs, targets, indices, num_points):
        """Classification loss """
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_logits']

        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.long, device=src_logits.device)
        target_classes_o = torch.cat([cls[J] for cls, (_, J) in zip(targets['gt_labels'], indices)])
        target_classes[idx] = target_classes_o

        loss_cls = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.class_weight)
        loss_dict = loss_cls

        return loss_dict

    def loss_mask(self, outputs, targets, indices, num_points):
        pred_masks = outputs['pred_masks']
        gt_masks = targets['gt_masks']

        loss_mask = self.focal_loss(pred_masks.squeeze(1), gt_masks)
        # loss_mask = self.focal_loss(pred_masks, gt_masks)
        loss_dict =  loss_mask

        # regularization
        # prior = torch.ones(args.num_class)/args.num_class
        # prior = prior.cuda()
        # pred_mean = torch.softmax(logits, dim=1).mean(0)
        # penalty = torch.sum(prior*torch.log(prior/pred_mean))

        return loss_dict

    @staticmethod
    def _get_src_permutation_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    def forward(
            self,
            outputs1, 
            targets1,
            epoch,

    ):
        num_points = sum(targets1['gt_nums'])
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=outputs1['pred_logits'].device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # losses1 = {}
        # loss_map1 = {
        #     'loss_reg': self.loss_reg,
        #     'loss_cls': self.loss_cls,
        #     'loss_mask': self.loss_mask
        # }

        indices = self.matcher(outputs1, targets1)
        # for loss_func in loss_map1.values():
        #     losses1.update(loss_func(outputs1, targets1, indices, num_points))

        # weight_dict = self.loss_weight

        #pred = pred.unsqueeze(1)
        #pred_iou = outputs2['pred_ious']
        #pred3 = outputs3['pred_masks'].unsqueeze(1)
        loss_dict = {
            'loss_reg': self.loss_reg(outputs1, targets1, indices, num_points) * 20,
            'loss_cls': self.loss_cls(outputs1, targets1, indices, num_points) * 20,
            'loss_mask': self.loss_mask(outputs1, targets1, indices, num_points) * 20,
            #'loss_sam2': self.lossfunc(pred,true2) * 10
            #'loss_dice_semantic': self.dice_loss(pred3, targets1['gt_masks'].unsqueeze(1)), #额外加的语义loss
            #'loss_iou': self.iou_loss(pred.unsqueeze(1), true2.float(), pred_iou)
        }
        # print(loss_dict)
        for k in loss_dict.keys():
            loss_dict[k] *= self.loss_weight[k](epoch)

        return loss_dict


def build_criterion(cfg,device):
    class_weight = torch.ones(cfg.data.num_classes + 1, dtype=torch.float).to(device)
    class_weight[-1] = cfg.criterion.eos_coef
    loss_weight = {
        'loss_focal':lambda epoch:  cfg.criterion.loss_focal,
        'loss_dice':lambda epoch:  cfg.criterion.loss_dice,
        'loss_iou': lambda epoch: cfg.criterion.loss_iou,
        'loss_dice_semantic': lambda epoch: cfg.criterion.loss_dice_semantic,


        'loss_cls': lambda epoch: cfg.criterion.cls_loss_coef,
        'loss_reg': lambda epoch: cfg.criterion.reg_loss_coef,
        'loss_mask': lambda epoch: cfg.criterion.mask_loss_coef,
        'loss_sam2': lambda epoch: cfg.criterion.sam2_loss_coef
    }
    matcher = build_matcher(cfg)

    criterion = Criterion(
        cfg.data.num_classes,
        matcher,
        class_weight=class_weight,
        loss_weight=loss_weight
    )

    return criterion , matcher
