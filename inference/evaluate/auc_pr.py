import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score

def auc_pr(y_true, abnormal_prob, save_path=None):
    '''
    Args:
        y_true(list): labels (0 is normal | 1 is abnormal)
        abnormal_prob(list): predicted probility of abnormal samples
    '''
    precision, recall, thresholds = precision_recall_curve(y_true, abnormal_prob)
    auc_pr = auc(recall, precision)
    avg_precision_sc = average_precision_score(y_true, abnormal_prob)

    print(f"[AUC PR]\t AUC-PR score: {auc_pr:.4f}")
    print(f"[AUC PR]\t AP     score: {avg_precision_sc:.4f}")

    plot_pr_curve(recall, precision, auc_pr, y_true, save_path)
    return auc_pr, precision, recall, thresholds

def plot_pr_curve(recall, precision, auc_pr, y_true, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR Curve (AUC = {auc_pr:.4f})")

    baseline = sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], color="navy", lw=2, linestyle="--", label=f"Baseline ({baseline:.4f})")

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("Recall (TPR)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.title("TKR Abnormal Detection PR Curve")
    plt.legend(loc="lower left")
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

    auc_pr(y_true, abnormal_prob)

if __name__ == "__main__":
    test()