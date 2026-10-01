import torch.nn as nn

LOSS_REGISTRY = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
}

def register_loss(name):
    """
    Decorator used to registered loss implementations.
    """
    def decorator(cls):
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered!")
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator