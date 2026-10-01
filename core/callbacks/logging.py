# callbacks/logging.py
import logging
from .base import BaseCallback

class LoggingCallback(BaseCallback):
    def __init__(self, log_file=None):
        self.logger = logging.getLogger("TKR_Trainer")
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)
        
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def on_epoch_end(self, trainer, epoch, metrics: dict, **kwargs):
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"Epoch [{epoch+1}] | {metrics_str}")

    def on_train_start(self, trainer, **kwargs):
        self.logger.info("Training Started.")

    def on_train_end(self, trainer, **kwargs):
        self.logger.info("Training Finished.")