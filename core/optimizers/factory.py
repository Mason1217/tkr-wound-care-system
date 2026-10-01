import torch
import torch.nn as nn
from omegaconf import DictConfig

from utils import filter_valid_args 
from .registry import OPTIMIZER_REGISTRY

def get_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """
    Args:
        model: Model need to be optimized
        cfg: Optimizer config (includes name, params)
    """
    name = cfg.name
    yaml_params = cfg.get("params", {}) 
    if yaml_params is None: yaml_params = {}

    # --- Default args ---
    defaults = {
        "lr": 1e-3, 
        "weight_decay": 0.0
    }
    
    if name == "AdamW":
        defaults.update({"weight_decay": 0.01})
    elif name == "SGD":
        defaults.update({"momentum": 0.9})

    # --- 2. Check Registry ---
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(OPTIMIZER_REGISTRY.keys())}")
    
    optim_cls = OPTIMIZER_REGISTRY[name]

    # Merge: YAML > Defaults
    all_params = {**defaults, **yaml_params}
    
    final_params = filter_valid_args(optim_cls, all_params)

    try:
        optimizer = optim_cls(model.parameters(), **final_params)
    except TypeError as e:
        raise TypeError(f"Error initializing optimizer {name}: {e}")

    return optimizer