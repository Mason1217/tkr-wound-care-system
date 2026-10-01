import os
import pandas as pd
import numpy as np
from .base import BaseCallback

class HistoryCallback(BaseCallback):
    def __init__(self, save_file: str):
        self.save_file = save_file
        self.history = []
        self.current_epoch_train_losses = []
        
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

    def on_batch_end(self, trainer, loss, **kwargs):
        """
        Collect training loss every batch.

        Args:
            loss: Tensor (detached)
        """
        if loss is not None:
            val = loss.item() if hasattr(loss, "item") else loss
            self.current_epoch_train_losses.append(val)

    def on_epoch_end(self, trainer, epoch, metrics: dict, **kwargs):
        """
        Save train loss (mean) & val metrics
        """
        avg_train_loss = 0.0
        if self.current_epoch_train_losses:
            avg_train_loss = np.mean(self.current_epoch_train_losses)
        
        record = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss
        }
        record.update(metrics)
        
        self.history.append(record)
        
        df = pd.DataFrame(self.history)
        df.to_csv(self.save_file, index=False)
        
        self.current_epoch_train_losses = []
