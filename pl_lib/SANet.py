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


class SANet(pl.LightningModule):
    def __init__(self, params) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model.SANet(params['channels'])

        self.loss_fn = bce_dice_loss

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        return self.model(x)

    @staticmethod
    def prob_correction(pred: Tensor) -> Tensor:
        pred[torch.where(pred > 0)] /= (pred > 0).float().mean()
        pred[torch.where(pred < 0)] /= (pred < 0).float().mean()

        return pred
    
    def configure_optimizers(self):
        optimizer = optim.SGD([{'params': self.model.res2net.parameters(), 'lr': self.hparams.params['lr'] * 0.1},
                                {'params': self.model.shal_attention.parameters(), 'lr': self.hparams.params['lr']}],
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

        pred = self.model(x)
        pred = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
        loss = self.loss_fn(pred, y)

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

        pred = self.prob_correction(self(x))
        pred = torch.sigmoid(pred)
        pred = F.interpolate(pred, size=origin_size, mode='bilinear', align_corners=True)

        return pred, file_name