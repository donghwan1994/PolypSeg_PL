import torch
import torch.nn as nn
import torch.nn.functional as F


def bce_iou_loss(pred: torch.Tensor, mask: torch.Tensor, 
                weights: torch.Tensor = None) -> torch.Tensor:
    if weights is not None:
        weights = weights
    else:
        weights = torch.ones(mask.size())
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weights * bce).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weights).sum(dim=(2, 3))
    union = ((pred + mask) * weights).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter +  1)

    return (wbce + wiou).mean()


def bce_dice_loss(pred: torch.Tensor, mask: torch.Tensor, 
                weights: torch.Tensor = None) -> torch.Tensor:
    if weights is not None:
        weights = weights
    else:
        weights = torch.ones(mask.size(), device=mask.device)
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weights * bce).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weights).sum(dim=(2, 3))
    union = ((pred + mask) * weights).sum(dim=(2, 3))
    wdice = 1 - (2 * inter) / (union +  1)

    return (wbce + wdice).mean()