import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import VGG, make_layers

os.environ['TORCH_HOME'] = 'weights'

from typing import *


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


# brought from ``https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49``.
def _vgg(arch: str, cfg: List[Union[str, int]], batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    if pretrained:
        model_state = torch.load('weights/vgg16-397923af.pth')
        model.load_state_dict(model_state)
    return model

def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("vgg16", [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], 
                False, pretrained, progress, **kwargs)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(vgg16(pretrained=True).features[:4].eval())
        blocks.append(vgg16(pretrained=True).features[4:9].eval())
        blocks.append(vgg16(pretrained=True).features[9:16].eval())
        blocks.append(vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        return loss