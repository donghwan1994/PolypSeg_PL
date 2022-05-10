from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PraNetLR(_LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        lr: float, 
        epoch: int,
        decay_rate: float = 0.1,
        decay_epoch: int = 30 
    ) -> None:
        self.lr = lr
        self.epoch = epoch
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch

        super(PraNetLR, self).__init__(optimizer, epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch)) for base_lr in
                    self.base_lrs]
        else:
            lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

        return lrs