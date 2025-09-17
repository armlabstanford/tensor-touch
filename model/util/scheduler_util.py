import torch.optim as optim

class LinearWarmupCosineAnnealingLR(optim.lr_scheduler.CosineAnnealingLR):
    def __init__(self, optimizer, warmup_steps, T_max, eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lrs = [base_lr * max(self.last_epoch, 1) / self.warmup_steps for base_lr in self.base_lrs]
            return lrs
        return super(LinearWarmupCosineAnnealingLR, self).get_lr()