import torch.optim as optim
from omegaconf import DictConfig

from .registry import SCHEDULER_REGISTRY
from .wrapper import SchedulerWrapper
from utils import filter_valid_args

def get_scheduler(optimizer: optim.Optimizer, cfg: DictConfig) -> SchedulerWrapper:
    """
    Args:
        optimizer: Initialized optimizer.
        cfg: Scheduler config (containing 'name' and 'params').
    Returns:
        wrapped_scheduler (SchedulerWrapper)
    """
    name = cfg.name
    yaml_params = cfg.get("params", {}) 
    if yaml_params is None: yaml_params = {}

    # --- 1. Define Default Parameters ---
    defaults = {}
    
    if name == "ReduceLROnPlateau":
        defaults = {
            "mode": "min",
            "factor": 0.1,
            "patience": 10
        }
    elif name == "CosineAnnealingLR":
        defaults = {
            "T_max": 100,
            "eta_min": 0.0
        }

    # --- 2. Check Registry ---
    if name not in SCHEDULER_REGISTRY:
        print(f"[Scheduler]\t Warning: Scheduler '{name}' not found. No scheduler will be used.")
        return None
    
    scheduler_cls = SCHEDULER_REGISTRY[name]

    # --- 3. Merge and Filter Parameters ---
    all_params = {**defaults, **yaml_params}
    final_params = filter_valid_args(scheduler_cls, all_params)

    # --- 4. Instantiation & Wrapping ---
    try:
        # Note: All PyTorch Schedulers take the optimizer as the first argument
        scheduler = scheduler_cls(optimizer, **final_params)
        
        # Return the wrapped object to provide a unified interface
        print(f"[Scheduler]\t {scheduler.__class__.__name__} params: {final_params}")
        return SchedulerWrapper(scheduler)
        
    except TypeError as e:
        raise TypeError(f"[Scheduler]\t Error initializing scheduler {name}: {e}")