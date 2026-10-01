import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import recall_score, confusion_matrix, fbeta_score

def optimize_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    '''
    Use LBFGS optimizer to find temperature (T) that minimizes cross entropy.
    '''
    device = logits.device 
    temperature = torch.nn.Parameter(torch.ones(1, device=device) * 1.5)
    
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=100, line_search_fn="strong_wolfe")

    def eval_closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits/temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_closure)

    initial_loss = F.cross_entropy(logits, labels).item()
    final_loss   = F.cross_entropy(logits/temperature, labels)

    print(f"[Calibrate]\t Initial NLL loss: {initial_loss:.4f} -> final NLL loss: {final_loss:.4f}")
    return temperature.item()

def find_best_threshold(calibrated_probs: np.ndarray, labels: np.ndarray, beta: float = 2.0) -> tuple[float, dict]:
    '''
    Find threshold (0.01 ~ 0.99) that meet best fbeta score.
    '''
    best_threshold = 0.5
    best_fbeta = 0.0
    best_metrics = {}

    thresholds = np.arange(0.01, 1.00, 0.01)
    for threshold in thresholds:

        preds   = (calibrated_probs[:, 1] >= threshold).astype(int)
        current_fbeta = fbeta_score(labels, preds, beta=beta, zero_division=0)
        recalls = recall_score(labels, preds, average=None, zero_division=0)

        normal_recall   = recalls[0] if isinstance(recalls, np.ndarray) else 0.0
        abnormal_recall = recalls[1] if isinstance(recalls, np.ndarray) else 0.0

        if current_fbeta > best_fbeta:
            best_fbeta = current_fbeta
            best_threshold = threshold
            best_metrics = {
                "fbeta": best_fbeta,
                "normal_recall": normal_recall,
                "abnormal_recall": abnormal_recall,
                "matrix": confusion_matrix(labels, preds).tolist(),
            }
        
    best_fbeta = best_metrics.get("fbeta")
    normal_recall = best_metrics.get("normal_recall")
    abnormal_recall = best_metrics.get("abnormal_recall")

    print(f"[Calibrate Result]\t f_beta: {best_fbeta} | normal_recall: {normal_recall} | abnormal recall: {abnormal_recall}")

    return float(best_threshold), best_metrics