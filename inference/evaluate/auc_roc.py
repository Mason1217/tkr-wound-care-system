import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def auc_roc(y_true, abnormal_prob, save_path=None):
    '''
    Args:
        y_true(list): labels (0 is normal | 1 is abnormal)
        abnormal_prob(list): predicted probility of abnormal samples
    '''
    fpr, tpr, thresholds = roc_curve(y_true, abnormal_prob)
    auc_roc = auc(fpr, tpr)

    print(f"[AUC ROC]\t AUC-ROC score: {auc_roc:.4f}")

    plot_roc_curve(fpr, tpr, auc_roc, save_path)

    return auc_roc, fpr, tpr, thresholds

def plot_roc_curve(fpr, tpr, auc_roc, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_roc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier (AUC = 0.5000)")

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("TKR Abnormal Detection ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def test():
    np.random.seed(42)
    normal_size   = 178
    abnormal_size = 35

    y_true_normal   = np.zeros(normal_size)
    y_true_abnormal = np.ones(abnormal_size)
    y_true = np.concatenate([y_true_normal, y_true_abnormal])

    prob_normal   = np.random.beta(a=1, b=5, size=normal_size)   # abnormal prob of normal samples
    prob_abnormal = np.random.beta(a=5, b=2, size=abnormal_size) # abnormal prob of abnormal samples
    abnormal_prob = np.concatenate([prob_normal, prob_abnormal])

    indices = np.arange(len(y_true))
    np.random.shuffle(indices)
    y_true = y_true[indices]
    abnormal_prob = abnormal_prob[indices]

    auc_roc(y_true, abnormal_prob)

if __name__ == "__main__":
    test()