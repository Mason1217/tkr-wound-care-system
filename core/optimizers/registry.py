import torch.optim as optim

OPTIMIZER_REGISTRY = {
    "AdamW": optim.AdamW,
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop
}

def register_optimizer(name):
    def decorator(cls):
        if name in OPTIMIZER_REGISTRY:
            raise ValueError(f"Optimizer '{name}' is already registered!")
        OPTIMIZER_REGISTRY[name] = cls
        return cls
    return decorator