import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .task import BaseTask

class Trainer:
    def __init__(
            self,
            callbacks: list = None,
            device: str | torch.device = "cuda"
    ):
        self.callbacks = callbacks or []
        self.device = device
        self.stop_training = False
        self.task = None

    def fit(
            self,
            task: BaseTask,
            train_loader: DataLoader,
            val_loader: DataLoader,
            max_epochs: int,
    ):
        self.task = task

        task.to(self.device)
        self._fire_callback("on_train_start")

        for epoch in range(max_epochs):
            self._fire_callback("on_epoch_start", epoch=epoch)

            self._run_train_epoch(task, train_loader, epoch, max_epochs)

            val_metrics = self.validate(task, val_loader)
            
            task.train_epoch_end()

            self._fire_callback("on_epoch_end", epoch=epoch, metrics=val_metrics)
            
            if self.stop_training:
                print(f"[Trainer]\t Early stopping triggered at epoch {epoch+1}")
                break
        
        self._fire_callback("on_train_end")

    def _run_train_epoch(
            self,
            task: BaseTask,
            loader: DataLoader,
            epoch: int,
            max_epochs: int
    ):

        task.train()
        train_pbar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{max_epochs} [Train]",
            leave=False,
        )

        for batch in train_pbar:
            # 1. Forward & Loss
            step_out = task.training_step(batch, self.device)
            
            # 2. Backward
            task.optimizer.zero_grad()
            step_out["loss"].backward()
                        
            task.optimizer.step()

            # 3. Logging & Callbacks
            loss_val = step_out["loss"].item()
            train_pbar.set_postfix(loss=f"{loss_val:.4f}")
            self._fire_callback("on_batch_end", loss=step_out["loss"].detach())

    def validate(self, task: BaseTask, dataloader: DataLoader) -> dict:

        task.eval()
        task.to(self.device)
        outputs_list = []
        
        with torch.inference_mode():
            val_pbar = tqdm(dataloader, desc="Validating", leave=False)
            for batch in val_pbar:
                step_out = task.val_step(batch, self.device)
                outputs_list.append(step_out)
        
        metrics = task.val_epoch_end(outputs_list)
        return metrics

    def test(self, task: BaseTask, dataloader: DataLoader) -> dict:
        self.task = task

        task.eval()
        task.to(self.device)
        outputs_list = []

        self._fire_callback("on_test_start")
        
        with torch.inference_mode():
            test_pbar = tqdm(dataloader, desc="Testing", leave=False)
            for batch in test_pbar:
                step_out = task.test_step(batch, self.device)
                outputs_list.append(step_out)
        
        metrics = task.test_epoch_end(outputs_list)
        
        self._fire_callback("on_test_end", metrics=metrics)
        return metrics

    def _fire_callback(self, event_name: str, **kwargs):
        for callback in self.callbacks:
            method = getattr(callback, event_name, None)
            if method:
                method(self, **kwargs)