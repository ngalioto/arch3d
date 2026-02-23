import math
from math import sqrt
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class ConstantWithWarmupLR(_LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        warmup_steps: int, 
        last_epoch: int = -1
    ) -> None:
        
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> torch.Tensor:
        if self._step_count < self.warmup_steps:
            return [base_lr * (self._step_count + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            return [base_lr for base_lr in self.base_lrs]

class SqrtDecayWithWarmupLR(_LRScheduler):
    
    """
    Follows the Attention is All You Need Paper
    """
    
    def __init__(
        self, 
        optimizer: Optimizer, 
        warmup_steps: int, 
        d_model: int, 
        target_lr: float = 2.4e-3,
        last_epoch: int = -1
    ) -> None:
        
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.lr_init = target_lr * sqrt(d_model * warmup_steps)
        super().__init__(optimizer, last_epoch)

    def _linear_warmup(self) -> torch.Tensor:
        """
        Linearly increases the learning rate over the course of the warmup period
        """
        return torch.tensor([(self._step_count + 1) / (sqrt(self.d_model) * self.warmup_steps ** 1.5)])
    
    def _lr_decay(self) -> torch.Tensor:
        """
        Exponentially decays the learning rate
        """
        return torch.tensor([1 / sqrt(self.d_model * self._step_count)])

    def get_lr(self) -> torch.Tensor:
        if self._step_count < self.warmup_steps:
            return self.lr_init * self._linear_warmup()
        else:
            return self.lr_init * self._lr_decay()

class CosineAnnealingWithWarmupLR(_LRScheduler):
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int,
        constant_steps: int,
        T_max: int = int(1e4), 
        min_lr: float = 1e-6, 
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.T_max = T_max
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Get the current epoch
        step = self._step_count

        if step < self.warmup_steps:
            # Linearly increase the learning rate during warmup
            warmup_factor = (step + 1) / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        elif step < self.constant_steps + self.warmup_steps:
            return [base_lr for base_lr in self.base_lrs]
        
        elif step <= self.T_max + self.warmup_steps:
            # Apply cosine annealing
            step -= self.warmup_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * (step - self.constant_steps) / (self.T_max - self.constant_steps)))
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]
        
        else: # Hold the learning rate constant at the min value after T_max + warmup steps
            return [self.min_lr for _ in self.base_lrs]

class CosineAnnealingWithWarmupLRandRestarts(_LRScheduler):
    def __init__(
        self, 
        optimizer, 
        warmup_steps: int,
        constant_steps: int,
        T_max: int = int(1e4), 
        min_lr: float = 1e-6, 
        last_epoch: int = -1,
        max_restarts: int | None = None
    ):
        
        if T_max <= constant_steps:
            raise ValueError(f'For cosine annealing, `T_max` must be greater than `constant_steps`, but got values {T_max=} and {constant_steps=}')
        
        self.warmup_steps = warmup_steps
        self.constant_steps = constant_steps
        self.T_max = T_max
        self.min_lr = min_lr
        self.max_restarts = torch.inf if max_restarts is None else max_restarts
        self.warmup_stage = True
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Get the current epoch
        step = self._step_count

        if self.warmup_stage:
            if step == self.warmup_steps:
                self.warmup_stage = False
            else:
                # Linearly increase the learning rate during warmup
                warmup_factor = (step + 1) / self.warmup_steps
                return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        if not self.warmup_stage:
            # Restart learning rate and remove warmup period
            step -= self.warmup_steps
            num_restarts = step // self.T_max
            step = step % self.T_max
            
            if num_restarts <= self.max_restarts:

                if step < self.constant_steps:
                    # Hold the learning rate constant at the max value
                    return [base_lr for base_lr in self.base_lrs]

                else:
                    # Apply cosine annealing
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * (step - self.constant_steps) / (self.T_max - self.constant_steps)))
                    return [self.min_lr + (base_lr - self.min_lr) * cosine_factor for base_lr in self.base_lrs]

            else:
                # Hold the learning rate constant at the min value
                return [self.min_lr for _ in self.base_lrs]
            

def configure_scheduler(
    config: dict,
    optimizer: Optimizer
) -> _LRScheduler:
    
    if config.optim['scheduler'] == 'inverse_sqrt':
        scheduler = {
            'scheduler': SqrtDecayWithWarmupLR(
                optimizer,
                warmup_steps=config.optim['warmup_steps'],
                d_model=config.hidden_size,
                target_lr = float(config.optim['lr'])
            ),
            'interval': 'step'
        }
    elif config.optim['scheduler'] == 'cosine_annealing':
        scheduler = {
            'scheduler': CosineAnnealingWithWarmupLR(
                optimizer,
                warmup_steps=config.optim['warmup_steps'],
                constant_steps=config.optim['constant_steps'],
                T_max=config.optim['T_max'],
                min_lr=float(config.optim['min_lr'])
            ),
            'interval': 'step'
        }
    elif config.optim['scheduler'] == 'cosine_annealing_with_restarts':
        scheduler = {
            'scheduler': CosineAnnealingWithWarmupLRandRestarts(
                optimizer,
                warmup_steps=config.optim['warmup_steps'],
                constant_steps=config.optim['constant_steps'],
                T_max=config.optim['T_max'],
                min_lr=float(config.optim['min_lr'])
            ),
            'interval': 'step'
        }
    elif config.optim['scheduler'] == 'constant_with_warmup':
        scheduler = {
            'scheduler': ConstantWithWarmupLR(
                optimizer,
                warmup_steps=config.optim['warmup_steps']
            ),
            'interval': 'step'
        }
    else:
        raise ValueError(f"Unsupported scheduler type: {config.optim['scheduler']}")

    return scheduler