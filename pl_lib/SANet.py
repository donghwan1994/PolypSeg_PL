import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
import pytorch_lightning as pl
from lib.res2net import res2net50_v1b_26w_4s
from lib.module import *
from lib.loss import *

from typing import *


class SANet(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)

        self.shal_attention = Shallow_Attention(self.hparams.params['channels'])

        self.loss_fn = bce_dice_loss

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)

        x1 = self.res2net.layer1(x)
        x2 = self.res2net.layer2(x1)
        x3 = self.res2net.layer3(x2)
        x4 = self.res2net.layer4(x3)

        pred = self.shal_attention(x2, x3, x4)

        return pred

    def prob_correction(self, pred: Tensor) -> Tensor:
        pred[torch.where(pred > 0)] /= (pred > 0).float().mean()
        pred[torch.where(pred < 0)] /= (pred < 0).float().mean()

        return pred
    
    def configure_optimizers(self):
        optimizer = optim.SGD([{'params': self.res2net.parameters(), 'lr': self.hparams.params['lr'] * 0.1},
                                {'params': self.shal_attention.parameters(), 'lr': self.hparams.params['lr']}],
                                momentum=self.hparams.params['momentum'],
                                weight_decay=self.hparams.params['weight_decay'],
                                nesterov=self.hparams.params['nesterov'])
        milestones = [i for i in range(self.hparams.params['milestones'][0], self.hparams.params['milestones'][1] + 1)]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=milestones,
                                                    gamma=self.hparams.params['lr_gamma'])
        return [optimizer], [lr_scheduler]

    def calculate_loss(self, batch, mode: str = "train") -> Tensor:
        x, y = batch

        map = self.forward(x)
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        loss = self.loss_fn(map, y)

        self.log(mode + "_loss", loss)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")
    
    def validation_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        x, _ = batch
        pred = self.forward(x)
        return self.prob_correction(pred)