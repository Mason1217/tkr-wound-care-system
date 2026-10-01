import numpy as np
from core import Trainer
from .base import BaseCallback

class EarlyStopping(BaseCallback):
    def __init__(self, monitor="val_loss", patience=5, mode="min", min_delta=0.0):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = np.inf if mode == "min" else -np.inf
        
    def on_epoch_end(self, trainer: Trainer, epoch, metrics: dict, **kwargs):
        current_score = metrics.get(self.monitor)
        
        if current_score is None:
            print(f"EarlyStopping Warning: Metric '{self.monitor}' not found in metrics.")
            return

        improvement = False
        if self.mode == "min":
            if current_score < (self.best_score - self.min_delta):
                improvement = True
        else: # mode == "max"
            if current_score > (self.best_score + self.min_delta):
                improvement = True

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.stop_training = True
                print(f"Early stopping triggered. Best {self.monitor}: {self.best_score:.4f}")