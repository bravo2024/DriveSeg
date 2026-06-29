from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        return self.weight_ce * self.ce(pred, target) + self.weight_dice * self.dice(pred, target)

def compute_seg_metrics(pred_masks, true_masks, num_classes):
    pred_masks = np.asarray(pred_masks)
    true_masks = np.asarray(true_masks)
    ious = []
    dice_scores = []
    pixel_acc = (pred_masks == true_masks).mean()
    for cls in range(num_classes):
        pred_cls = pred_masks == cls
        true_cls = true_masks == cls
        intersection = (pred_cls & true_cls).sum()
        union = (pred_cls | true_cls).sum()
        if union == 0:
            ious.append(float("nan"))
            dice_scores.append(float("nan"))
        else:
            ious.append(intersection / union)
            dice_scores.append(2 * intersection / (pred_cls.sum() + true_cls.sum() + 1e-6))
    miou = np.nanmean(ious)
    mdice = np.nanmean(dice_scores)
    return {
        "miou": float(miou),
        "dice": float(mdice),
        "pixel_acc": float(pixel_acc),
        "per_class_iou": [float(v) if not np.isnan(v) else 0.0 for v in ious],
        "per_class_dice": [float(v) if not np.isnan(v) else 0.0 for v in dice_scores],
    }
