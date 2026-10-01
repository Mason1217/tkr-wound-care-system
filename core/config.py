import os
import torch
from omegaconf import OmegaConf, DictConfig

class ConfigManager:
    def __init__(self, config_path: str = None, config_dict: dict = None):
        """
        Args:
            config_path: YAML filepath
            config_dict: For testing
        """
        if config_path:
            self._cfg = OmegaConf.load(config_path)
        elif config_dict:
            self._cfg = OmegaConf.create(config_dict)
        else:
            self._cfg = OmegaConf.create({})

        self._resolve_device()

    @property
    def cfg(self) -> DictConfig:
        return self._cfg

    def merge_with(self, other_cfg: dict | str):
        """
        Merge with base config.

        Args:
            other_cfg: Can be another config file or dict (ex: new_stage.yaml)
        """
        if isinstance(other_cfg, str):
            other = OmegaConf.load(other_cfg)
        else:
            other = OmegaConf.create(other_cfg)
        
        self._cfg = OmegaConf.merge(self._cfg, other)
        self.resolve_paths()
        self._resolve_device()

    def resolve_paths(self):
        """
        Convert to absolute path.
        """
        root = os.getcwd()

        path_keys = [
            "output_dir",
            "data.metadata_dir",
            "data.image_dir",
            "data.files.image_id_map_original",
            "data.files.image_id_map_cropped",
        ]
        
        for key in path_keys:
            original_path = OmegaConf.select(self._cfg, key)
            
            if original_path and not os.path.isabs(original_path):
                abs_path = os.path.join(root, original_path)
                OmegaConf.update(self._cfg, key, abs_path)      
    
    def _resolve_device(self):
        """
        Detect available hardware and update 'device' field if it is set to 'auto'.
        Priority: CUDA > MPS > CPU
        """
        current_device = self._cfg.get("device", "cpu")

        if current_device == "auto":
            if torch.cuda.is_available():
                resolved = "cuda"
                print(f"[Config]\t Device 'auto' resolved to: CUDA ({torch.cuda.get_device_name(0)})")
            elif torch.backends.mps.is_available():
                resolved = "mps"
                print("[Config]\t Device 'auto' resolved to: MPS (Apple Silicon)")
            else:
                resolved = "cpu"
                print("[Config]\t Device 'auto' resolved to: CPU")
            
            OmegaConf.update(self._cfg, "device", resolved)
            
    def save(self, save_path: str):
        """
        Dump current config (usually on the beginning of training)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        OmegaConf.save(self._cfg, save_path)

    def __getattr__(self, name):
        return getattr(self._cfg, name)