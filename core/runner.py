import os
import itertools
import copy
from typing import List

import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig

from .models import get_model
from .loss_functions import get_loss_fn
from .optimizers import get_optimizer
from .schedulers import get_scheduler
from .mix import get_mix_fn

from .callbacks.checkpoint import ModelCheckpoint
from .callbacks.early_stopping import EarlyStopping
from .callbacks.logging import LoggingCallback
from .callbacks.history import HistoryCallback


from .config import ConfigManager
from .data.datamodule import TKRDataModule
from .trainer import Trainer
from .task import TKRTask

class TKRTrainingRunner:
    def __init__(
            self,
            base_config_path: str,
            exp_config_path: str = None,
    ):
        # 1. Initialize config manager
        self.cfg_manager = ConfigManager(base_config_path)
        
        # 2. Merge experiment config
        if exp_config_path:
            self.cfg_manager.merge_with(exp_config_path)
        
        self.cfg_manager.resolve_paths()

        # 3. Data module
        self.datamodule = TKRDataModule(self.cfg_manager.cfg)

        # Setup output directory
        self.root_output_dir = os.path.join(
            self.cfg_manager.cfg.output_dir, 
            self.cfg_manager.cfg.exp_name
        )
        os.makedirs(self.root_output_dir, exist_ok=True)

    def run(self):
        """
        Main entry, determine single job or sweep.
        """
        base_cfg = self.cfg_manager.cfg
        sweep_config = base_cfg.get("sweep", None)

        if not sweep_config:
            # --- Case A: Single job ---
            print(f"\n=== Starting Single Run: {base_cfg.exp_name} ===\n")
            self._run_single_job(base_cfg, run_dir=self.root_output_dir)
        else:
            # --- Case B: Sweep ---
            self._run_sweep_job(base_cfg, sweep_config)

    def _run_sweep_job(self, base_cfg: DictConfig, sweep_config: DictConfig):
        # Prepare parameter combinations
        param_keys = list(sweep_config.keys())
        param_values = list(sweep_config.values())
        combinations = list(itertools.product(*param_values))
        
        total_runs = len(combinations)
        print(f"\n[Runner]\t [Sweep Detected] Found {len(param_keys)} parameters, generating {total_runs} runs.\n")

        for i, combo in enumerate(combinations):
            run_cfg = copy.deepcopy(base_cfg)
            
            run_name = self._generate_run_name(i + 1, param_keys, combo)
            run_dir = os.path.join(self.root_output_dir, run_name)
            
            print(f"\n=== [{i+1}/{total_runs}] Starting Sweep Run: {run_name} ===\n")
            
            # Override parameters
            for key, value in zip(param_keys, combo):
                OmegaConf.update(run_cfg, key, value)
            
            self._run_single_job(run_cfg, run_dir)

    def _run_single_job(self, cfg: DictConfig, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))

        print(f"[Runner]\t > Re-initializing DataModule with new config...")
        run_specific_cfg = copy.deepcopy(cfg)
        run_specific_cfg.output_dir = run_dir 
        
        self.datamodule.cfg = run_specific_cfg
        self.datamodule.data_cfg = run_specific_cfg.data
        
        self.datamodule.setup()
        self.cls_freq_list = self.datamodule.get_cls_freq_list()

        stages = cfg.get("stages", [])
        if not stages:
            print("[Runner]\t Warning: No stages defined in config.")
            return

        current_model = None
        
        for stage_idx, stage_cfg in enumerate(stages):
            print(f"[Runner]\t > Running Stage {stage_idx+1}: {stage_cfg.name}")
            
            current_stage_full_cfg = self._merge_stage_config(cfg, stage_cfg)
            
            if current_model is None:
                print(f"[Runner]\t Creating model: {current_stage_full_cfg.model.name}")
                current_model = get_model(current_stage_full_cfg.model)
                current_model = current_model.to(current_stage_full_cfg.device)
            else:
                print(f"[Runner]\t Resuming model from previous stage...")
            
            freeze_layers = stage_cfg.get("freeze", [])
            unfreeze_layers = stage_cfg.get("unfreeze", [])
            self._apply_freezing(current_model, freeze_layers, unfreeze_layers)

            optimizer = get_optimizer(current_model, current_stage_full_cfg.optimizer)
            scheduler = get_scheduler(optimizer, current_stage_full_cfg.scheduler)
            
            loss_fn = get_loss_fn(
                current_stage_full_cfg.loss, 
                device=current_stage_full_cfg.device,
                cls_freq_list=self.cls_freq_list,
            )

            mixup_fn = get_mix_fn(current_stage_full_cfg)

            task = TKRTask(
                model=current_model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                scheduler=scheduler,
                mixup_fn = mixup_fn,
            )

            callbacks = self._setup_callbacks(current_stage_full_cfg, run_dir, stage_cfg.name)

            trainer = Trainer(
                callbacks=callbacks,
                device=current_stage_full_cfg.device
            )
            
            train_dataloader, val_dataloader = self._get_dataloaders(current_stage_full_cfg)

            trainer.fit(
                task=task,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                max_epochs=stage_cfg.epochs,
            )

    def _get_dataloaders(self, cfg):
        current_bs = cfg.data.batch_size

        train_dataloader = self.datamodule.get_train_loader(batch_size=current_bs)
        val_dataloader = self.datamodule.get_val_loader(batch_size=current_bs)

        run_overfitting_test = cfg.get("debug", {}).get("overfit_single_batch", False)
        if run_overfitting_test:
            print("\n" + "="*50, "🚨 [DEBUG MODE] OVERFIT SINGLE BATCH 🚨", "="*50)

            if not hasattr(self, "debug_single_batch") or self.debug_single_batch is None:
                self.debug_single_batch = next(iter(train_dataloader))

            train_dataloader = [self.debug_single_batch]
            val_dataloader = [self.debug_single_batch]
        
        return train_dataloader, val_dataloader

    def _setup_callbacks(self, cfg: DictConfig, run_dir: str, stage_name: str) -> List:
        callbacks_list = []
        cb_cfg = cfg.get("callbacks", {})
        
        # Checkpoint
        if cb_cfg.get("checkpoint"):
            save_dir = os.path.join(run_dir, "checkpoints", stage_name)
            callbacks_list.append(ModelCheckpoint(
                save_dir=save_dir,
                monitor=cb_cfg.checkpoint.get("monitor", "val_loss"),
                mode=cb_cfg.checkpoint.get("mode", "min"),
                save_best_only=cb_cfg.checkpoint.get("save_best_only", True),
            ))

        # Early Stopping
        if cb_cfg.get("early_stopping", {}).get("enable"):
            es_cfg = cb_cfg.early_stopping
            callbacks_list.append(EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=es_cfg.get("patience", 10),
                mode=es_cfg.get("mode", "min"),
                min_delta=es_cfg.get("min_delta", 0),
            ))
            
        # Logging
        callbacks_list.append(LoggingCallback(log_file=os.path.join(run_dir, f"{stage_name}.log")))

        # History
        csv_path = os.path.join(run_dir, f"{stage_name}_history.csv")
        callbacks_list.append(HistoryCallback(save_file=csv_path))

        return callbacks_list

    def _apply_freezing(self, model: nn.Module, layers_to_freeze: List[str], layers_to_unfreeze: List[str] = None):

        if layers_to_freeze is None and layers_to_unfreeze is None:
            print("[Runner]\t [Freezing] No config provided (None). Keeping model's original state (e.g., for LoRA).")
            return
        
        if layers_to_freeze is not None:
            if len(layers_to_freeze) == 0:
                print("[Runner]\t [Freezing] Empty list provided. Forcing UNFREEZE ALL layers.")
                for param in model.parameters():
                    param.requires_grad = True
                return

            print(f"[Runner]\t Freezing layers containing: {layers_to_freeze}")
            for param in model.parameters():
                param.requires_grad = True
            
            frozen_count = 0
            for name, param in model.named_parameters():
                if any(keyword in name for keyword in layers_to_freeze):
                    param.requires_grad = False
                    frozen_count += 1
                    # print(f"{name}->{param.requires_grad}")
            print(f"[Runner]\t -> Frozen {frozen_count} parameters.")
        
        if layers_to_unfreeze:
            print(f"[Runner]\t [Unfreezing] Re-enabling layers containing: {layers_to_unfreeze}")
            unfrozen_count = 0
            for name, param in model.named_parameters():
                if any(keyword in name for keyword in layers_to_unfreeze):
                    param.requires_grad = True
                    unfrozen_count += 1
            print(f"[Runner]\t -> Unfrozen {unfrozen_count} parameters specificially.")

        # For checking...
        print("\n", "="*50, "\n")
        for name, param in model.named_parameters():
            print(f"[Runner]\t {name:<100}\t->\t{param.requires_grad:>5}")
        print("\n", "="*50, "\n")
    
    def _generate_run_name(self, run_index: int, params: list, values: list) -> str:
        """
        Generate run name that is easy to read.
        ex: run1_ViTB16_FocalLoss_lr1e-05
        """
        parts = [f"run{run_index}"]
        
        for k, v in zip(params, values):
            short_key = k.split('.')[-1]
            
            if short_key == "name": 
                parts.append(str(v))
            elif short_key == "lr":
                parts.append(f"lr{v}")
            elif short_key == "stages":
                if isinstance(v, list):
                    count = len(v)
                    parts.append(f"{count}Stage")
            else:
                val_str = str(v)
                if len(val_str) > 20: 
                    parts.append(f"{short_key}_complex")
                else:
                    parts.append(f"{short_key}{v}")
                
        return "_".join(parts)
    
    def _merge_stage_config(self, base_cfg: DictConfig, stage_cfg: DictConfig) -> DictConfig:
        """
        Merge stage config to base config.
        """
        new_cfg = copy.deepcopy(base_cfg)

        if stage_cfg.get("override_lr") is not None:
            if "optimizer" in new_cfg and "params" in new_cfg.optimizer:
                new_cfg.optimizer.params.lr = stage_cfg.override_lr

        # Replace
        replace_keys = ["loss", "scheduler"]
        
        # Merge
        merge_keys = ["mixup", "callbacks"]
        
        for key in replace_keys:
            if stage_cfg.get(key):
                new_cfg[key] = stage_cfg[key]
                
        for key in merge_keys:
            if stage_cfg.get(key):
                if key in new_cfg:
                    new_cfg[key] = OmegaConf.merge(new_cfg[key], stage_cfg[key])
                else:
                    new_cfg[key] = stage_cfg[key]

        return new_cfg