import torch
from torch import Tensor
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F

from lib.loss import *
from lib.module import *
from lib.res2net import res2net50_v1b_26w_4s
import lib.model as model

from typing import *

class PraNet(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.scales = [0.75, 1.25, 1.]
        self.model = model.PraNet(params['channels'])

        self.loss_fn = bce_iou_loss

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return self.model(x) # pred2, pred3, pred4, pred5

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.params['lr'])
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                step_size=self.hparams.params['decay_epoch'],
                                                gamma=self.hparams.params['decay_rate'])
        return [optimizer], [lr_scheduler]

    def calculate_loss(self, batch, mode: str = "train") -> Tensor:
        x, y = batch

        for scale in self.scales:
            base_size = int(round(x.shape[-2] * scale / 32) * 32), int(round(x.shape[-1] * scale / 32) * 32)
            x = F.interpolate(x, size=base_size, mode='bilinear', align_corners=True)
            y = F.interpolate(y, size=base_size, mode='bilinear', align_corners=True)
            pred2, pred3, pred4, pred5 = self.forward(x)
            
            pred2 = F.interpolate(pred2, size=base_size, mode='bilinear', align_corners=False)
            pred3 = F.interpolate(pred3, size=base_size, mode='bilinear', align_corners=False)
            pred4 = F.interpolate(pred4, size=base_size, mode='bilinear', align_corners=False)
            pred5 = F.interpolate(pred5, size=base_size, mode='bilinear', align_corners=False)

            weights = 1 + 5 * torch.abs(F.avg_pool2d(y, kernel_size=31, stride=1, padding=15) - y)
            loss2 = self.loss_fn(pred2, y, weights=weights)
            loss3 = self.loss_fn(pred3, y, weights=weights)
            loss4 = self.loss_fn(pred4, y, weights=weights)
            loss5 = self.loss_fn(pred5, y, weights=weights)
            loss = loss2 + loss3 + loss4 + loss5

            if scale == 1:
                self.log(mode + "_loss2", loss2)
                self.log(mode + "_loss3", loss3)
                self.log(mode + "_loss4", loss4)
                self.log(mode + "_loss5", loss5)
                self.log(mode + "_total_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="train")
    
    def validation_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode="val")
