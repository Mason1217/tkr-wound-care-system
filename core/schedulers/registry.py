import torch.optim.lr_scheduler as lr_scheduler

SCHEDULER_REGISTRY = {
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "StepLR": lr_scheduler.StepLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "LinearLR": lr_scheduler.LinearLR,
}

def register_scheduler(name):
    def decorator(cls):
        if name in SCHEDULER_REGISTRY:
            raise ValueError(f"Scheduler '{name}' is already registered!")
        SCHEDULER_REGISTRY[name] = cls
        return cls
    return decorator