# callbacks/checkpoint.py
import os
import torch
import numpy as np
from core import Trainer
from .base import BaseCallback

class ModelCheckpoint(BaseCallback):
    def __init__(self, save_dir, monitor="val_loss", mode="min", save_best_only=True):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = np.inf if mode == "min" else -np.inf
        
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer: Trainer, epoch, metrics: dict, **kwargs):
        current_score = metrics.get(self.monitor)
        
        # 1. Save Best Model
        if current_score is not None:
            improvement = (current_score < self.best_score) if self.mode == "min" else (current_score > self.best_score)
            
            if improvement:
                self.best_score = current_score
                self._save_checkpoint(trainer, "best_model.pth")
                print(f"New best model saved with {self.monitor}: {self.best_score:.4f}")
        else:
            print(f"ModelCheckpoint Warning: Metric '{self.monitor}' not found in metrics.")

        # 2. Save Regular Checkpoint (if not only best)
        if not self.save_best_only:
             self._save_checkpoint(trainer, f"epoch_{epoch+1}.pth")

    def on_train_end(self, trainer: Trainer, **kwargs):
        self._save_checkpoint(trainer, "final_model.pth")
        print("Final model saved.")

    def _save_checkpoint(self, trainer: Trainer, filename):
        path = os.path.join(self.save_dir, filename)

        state = {
            'model_state_dict': trainer.task.model.state_dict(),
            'optimizer_state_dict': trainer.task.optimizer.state_dict(),
        }
        torch.save(state, path)