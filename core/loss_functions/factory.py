import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from utils import filter_valid_args
from .registry import LOSS_REGISTRY
from .focal_loss import FocalLoss
from .balanced_bce import BalancedBCELoss

def get_loss_fn(cfg: DictConfig, device="cpu", **runtime_args) -> nn.Module:
    """
    Args:
        cfg: Loss config (includes name, params)
        device: 
        **runtime_args: (ex: cls_freq_list)
    """
    name = cfg.name

    raw_params = cfg.get("params", {})
    if raw_params is None:
        yaml_params = {}
    elif isinstance(raw_params, DictConfig):
        yaml_params = OmegaConf.to_container(raw_params, resolve=True)
    else:
        yaml_params = dict(raw_params)

    # --- Defaults params ---
    defaults = {"device": device}

    if name == "CrossEntropyLoss":
        if "device" in defaults: del defaults["device"]
        if yaml_params.get("use_weight") and runtime_args.get("cls_freq_list") is not None:
            weights = _cal_inv_class_freq(runtime_args.get("cls_freq_list"))
            yaml_params["weight"] = torch.tensor(weights, dtype=torch.float32)
            print(f"[Loss]\t [CrossEntropyLoss] Auto-computed class weights: {weights}")

    elif name == "BalancedBCELoss":
        defaults.update({
            "tau": 1.0, 
            "reduction": "mean",
            "cls_freq_list": None
        })
        
    elif name == "FocalLoss":
        defaults.update({
            "gamma": 2.0, 
            "alpha": 0.25, 
            "reduction": "mean"
        })
        
    # --- Check LOSS_REGISTRY ---
    if name in LOSS_REGISTRY:
        loss_cls = LOSS_REGISTRY[name]
    else:
        raise ValueError(f"Unknown loss function: {name}. Available: {list(LOSS_REGISTRY.keys()) + ['CrossEntropyLoss']}")
    
    loss_cls = LOSS_REGISTRY[name]

    # Priority: YAML > Defaults (YAML would override Defaults)
    all_params = {**defaults, **yaml_params, **runtime_args}
    filtered_params = filter_valid_args(loss_cls, all_params)

    print(f"[Loss]\t {loss_cls.__name__} param: {filtered_params}")

    try:
        return loss_cls(**filtered_params)
    except TypeError as e:
        raise TypeError(f"Error initializing {name}: {e}. \nProvided valid params: {filtered_params.keys()}")
    
def _cal_inv_class_freq(cls_freq_list: list) -> list:
    weights = []

    for f in cls_freq_list:
        if f > 0:
            weights.append(1.0 / f)
        else:
            weights.append(0.0)
    
    weight_sum = sum(weights)
    if weight_sum > 0:
        weights = [w / weight_sum for w in weights]
    
    return weights