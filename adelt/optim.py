import torch


class InverseSquareRootSchedule(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_updates, last_epoch=-1):
        lr_step = 1.0 / warmup_updates
        decay_factor = warmup_updates ** 0.5

        def lr_lambda(num_updates):
            if num_updates < warmup_updates:
                return num_updates * lr_step
            else:
                return decay_factor * (num_updates ** -0.5)

        super(InverseSquareRootSchedule, self).__init__(optimizer, lr_lambda, last_epoch)
