from .mixup import Mixup
from .cutmix import CutMix

def get_mix_fn(cfg):

        mixup_cfg = cfg.get("mixup", {})
        mix_fn = None

        if mixup_cfg.get("enable", False):
                aug_type = mixup_cfg.get("type", "mixup").lower()
                print(f"[Mix]\t Enabling {aug_type.capitalize()} (alpha={mixup_cfg.alpha}, prob={mixup_cfg.prob}, num_class={cfg.model.params.num_classes})")
                
                if aug_type == "mixup":
                    mix_fn = Mixup(
                        alpha=mixup_cfg.alpha,
                        prob=mixup_cfg.prob,
                        num_classes=cfg.model.params.num_classes,
                    )
                elif aug_type == "cutmix":
                    mix_fn = CutMix(
                        alpha=mixup_cfg.alpha,
                        prob=mixup_cfg.prob,
                        num_classes=cfg.model.params.num_classes,
                    )
                else:
                    raise ValueError(f"Unknown data augmentation type in config: {aug_type}")
        
        return mix_fn