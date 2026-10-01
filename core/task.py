import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from .schedulers import SchedulerWrapper

from .mix import BaseMixAugment

class BaseTask(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
            scheduler: SchedulerWrapper = None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler

    def training_step(self, batch, device):
        '''
        Process a single training batch.
        Returns:
            step_out(dict): { "loss": Tensor, "preds": Tensor, "targets": Tensor, ... }
        '''
        raise NotImplementedError

    def val_step(self, batch, device):
        '''
        Process a single validation batch.
        Returns:
            step_out(dict): { "loss": Tensor, "preds": Tensor, "targets": Tensor, ... }
        '''
        raise NotImplementedError

    def val_epoch_end(self, outputs: list) -> dict:
        '''
        Calculate metrics for the whole validation epoch.
        Args:
            outputs (list): List of outputs from val_step.
        Returns:
            result(dict): { "val_loss": float, "val_acc": float, ... }
        '''
        raise NotImplementedError

    def train_epoch_end(self):
        if self.scheduler:
            self.scheduler.step()

    def test_step(self, batch, device):
        raise NotImplementedError

    def test_epoch_end(self, outputs: list) -> dict:
        raise NotImplementedError

class TKRTask(BaseTask):
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn: nn.Module,
            scheduler: SchedulerWrapper = None,
            mixup_fn: BaseMixAugment = None,
    ):
        super().__init__(model, optimizer, loss_fn, scheduler)
        self.mixup_fn = mixup_fn
    
    def training_step(
            self,
            batch: tuple | list,
            device: torch.device | str,
    ) -> dict:
        images, metadata, labels, _ = batch
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        
        if self.mixup_fn is not None:
            images, metadata, labels = self.mixup_fn(images, metadata, labels)

        outputs = self.model(images, metadata)
        loss = self.loss_fn(outputs, labels)
        
        return {"loss": loss, "preds": outputs, "targets": labels}

    def val_step(
            self,
            batch: tuple | list,
            device: torch.device | str,
    ) -> dict:
        images, metadata, labels, _ = batch
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
        
        outputs = self.model(images, metadata)
        loss = self.loss_fn(outputs, labels)
        
        return {
            "loss": loss.detach().cpu(),
            "preds": outputs.detach().cpu(),
            "targets": labels.detach().cpu(),
        }

    def val_epoch_end(self, outputs: list) -> dict:
        '''
        Returns:
            metrics(dict):
                {"val_loss":float, "val_acc":float, "val_macro_recall":float,
                 "val_recalls": list | float}
        '''
        val_loss = torch.stack([x["loss"] for x in outputs]).mean().item()
        
        all_preds = torch.cat([x["preds"] for x in outputs], dim=0)
        all_targets = torch.cat([x["targets"] for x in outputs], dim=0)
        
        pred_indices = torch.argmax(all_preds, dim=1)
        
        if all_targets.dim() > 1:
            target_indices = torch.argmax(all_targets, dim=1)
        else:
            target_indices = all_targets

        y_true = target_indices.cpu().numpy()
        y_pred = pred_indices.cpu().numpy()

        # Accuracy
        acc = (y_pred == y_true).mean() * 100

        # Recall / Precision / F1
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0, average=None
        )
        macro_recall = np.mean(rec) * 100

        return {
            "val_loss": val_loss,
            "val_acc": acc,
            "val_macro_recall": macro_recall,
            "val_recalls": rec,
            "cur_lr": self.optimizer.param_groups[0]["lr"],
        }
    
    def test_step(self, batch, device) -> dict:
        images, metadata, labels, sample_ids = batch
        images, metadata, labels = images.to(device), metadata.to(device), labels.to(device) # labels 也要轉 device
        
        outputs = self.model(images, metadata)
        loss = self.loss_fn(outputs, labels)
        
        return {
            "loss": loss,
            "preds": outputs.detach(), 
            "targets": labels.detach(),
            "sample_ids": sample_ids
        }

    def test_epoch_end(self, outputs: list) -> dict:
        metrics = self.val_epoch_end(outputs)
        
        print(f"[Task]\t Test Metrics: {metrics}")
        return metrics