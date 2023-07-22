from torch.optim.lr_scheduler import _LRScheduler


class StepLRDecay(_LRScheduler):
    def __init__(self, optimizer, max_decay_steps, steps=(0.8, 0.9), scales=(0.1, 0.1)):
        assert max_decay_steps > 1., 'max_decay_steps should be greater than 1.'
        assert len(steps) == len(scales)

        self.max_decay_steps = max_decay_steps
        self.steps = steps
        self.scales = scales

        self.last_step = 0

        super().__init__(optimizer)

    def get_lr(self):
        factor = 1

        for step, scale in zip(self.steps, self.scales):
            if self.last_step >= step * self.max_decay_steps:
                factor *= scale

        return [base_lr * factor for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1

        self.last_step = step if step != 0 else 1

        if self.last_step <= self.max_decay_steps:
            decay_lrs = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
