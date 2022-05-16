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


class MSNet(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model.MSNet(params['channels'])

        self.bce_iou_loss = bce_iou_loss
        self.loss_net = VGGPerceptualLoss()
        self.loss_net.eval()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

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
        lr_lambda = lambda epoch: (1 - abs(epoch / (self.hparams.params['epoch'] + 1) * 2 - 1))
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def calculate_loss(self, batch, mode: str = "train") -> Tensor:
        x, y = batch

        pred = self.model(x)
        pred = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=False)

        weights = 1 + 5 * torch.abs(F.avg_pool2d(y, kernel_size=31, stride=1, padding=15) - y)
        
        loss = self.bce_iou_loss(pred, y, weights=weights) + \
                self.loss_net(torch.sigmoid(pred), y) * self.hparams.params['coeff']

        self.log(mode + "_loss", loss, batch_size=self.hparams.params['batch_size'])

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