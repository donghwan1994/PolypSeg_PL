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

class Prototype(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.scales = [0.75, 1.25, 1.]
        self.model = model.Protytpe_model(params['channels'])

        self.seg_loss = bce_iou_loss
        self.edge_loss = edge_loss

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        _, pred, _, _, _ = self.model(x) # pred2, pred3, pred4, pred5
        return pred

    def configure_optimizers(self):
        base, head = [], []
        for name, param in self.model.named_parameters():
            if 'res2net' in name:
                if 'conv1' in name or 'bn1' in name:
                    pass
                else:
                    base.append(param)
            else:
                head.append(param)
        optimizer = optim.SGD([{'params': base, 'lr': self.hparams.params['lr'] * 0.1},
                                {'params': head, 'lr': self.hparams.params['lr']}],
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

        # for scale in self.scales:
        # base_size = int(round(x.shape[-2] * scale / 32) * 32), int(round(x.shape[-1] * scale / 32) * 32)
        # x = F.interpolate(x, size=base_size, mode='bilinear', align_corners=True)
        # y = F.interpolate(y, size=base_size, mode='bilinear', align_corners=True)
        edge, pred2, pred3, pred4, pred5 = self.model(x)
        
        edge = F.interpolate(edge, size=x.shape[-2:], mode='bilinear', align_corners=False)
        pred2 = F.interpolate(pred2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        pred3 = F.interpolate(pred3, size=x.shape[-2:], mode='bilinear', align_corners=False)
        pred4 = F.interpolate(pred4, size=x.shape[-2:], mode='bilinear', align_corners=False)
        pred5 = F.interpolate(pred5, size=x.shape[-2:], mode='bilinear', align_corners=False)

        weights = 1 + 5 * torch.abs(F.avg_pool2d(y, kernel_size=31, stride=1, padding=15) - y)
        edge_loss = self.edge_loss(edge, y)
        loss2 = self.seg_loss(pred2, y, weights=weights)
        loss3 = self.seg_loss(pred3, y, weights=weights)
        loss4 = self.seg_loss(pred4, y, weights=weights)
        loss5 = self.seg_loss(pred5, y, weights=weights)
        loss = loss2 + loss3 + loss4 + loss5 + edge_loss * self.hparams.params['trade_off']

        # if scale == 1:
        self.log(mode + "_loss2", loss2, batch_size=self.hparams.params['batch_size'])
        self.log(mode + "_loss3", loss3, batch_size=self.hparams.params['batch_size'])
        self.log(mode + "_loss4", loss4, batch_size=self.hparams.params['batch_size'])
        self.log(mode + "_loss5", loss5, batch_size=self.hparams.params['batch_size'])
        self.log(mode + "_total_loss", loss, batch_size=self.hparams.params['batch_size'])

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        return self.calculate_loss(batch['data'], mode="train")
    
    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        return self.calculate_loss(batch['data'], mode="val")

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, _ = batch['data']
        file_name = batch['file_name']
        origin_size = batch['origin_size']

        pred = torch.sigmoid(self(x))
        pred = F.interpolate(pred, size=origin_size, mode='bilinear', align_corners=True)

        return pred, file_name