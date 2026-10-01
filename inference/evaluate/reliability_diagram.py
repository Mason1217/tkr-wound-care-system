import numpy as np
import matplotlib.pyplot as plt

def plot_reliability_diagram(y_true, y_pred, y_prob, n_bins=10, save_path= None) -> float:
    '''
    Plot reliability diagram from "On Calibration of Modern Neural Networks"

    Args:
        y_true(list): labels
        y_pred(list): model predictions
        y_prob(list): model's raw output (after softmax)
        n_bins(int) : number of bins in diagram (usually 10 ~ 15)

    Returns:
        ECE(float): expected calibration error
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    corrects = (y_true == y_pred).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins, right=True) - 1 # digitize return start from 1 => need to -1 (index start from 0)

    bin_acc   = np.zeros(n_bins)
    bin_conf  = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (bin_indices == b)

        if np.any(mask):
            bin_acc[b]   = np.mean(corrects[mask])
            bin_conf[b]  = np.mean(y_prob[mask])
            bin_count[b] = np.sum(mask)
    
            print(f"[Reliability Diagram]\t bin_{b} count: {bin_count[b]:10.2f} | accuracy: {bin_acc[b]:4.2f}")
    
    print(f"[Reliability Diagram]\t Total samples: {len(corrects)} | Correct samples: {np.sum(corrects)}")

    n_total = len(y_prob)
    ece = np.sum(np.abs(bin_acc - bin_conf) * (bin_count / n_total))
    
    _plot(bins, n_bins, bin_count, bin_acc, bin_conf, ece, save_path)

    return ece


def _plot(bins, n_bins, bin_count, bin_acc, bin_conf, ece, save_path=None):
    
    fig, ax = plt.subplots(figsize=(6, 6))

    bin_centers = (bins[: -1] + bins[1: ]) / 2
    width = 1.0 / n_bins

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    labeled_higher_conf = False
    labeled_higher_acc  = False
    labeled_lower_conf  = False
    labeled_lower_acc   = False

    for i in range(n_bins):
        if bin_count[i] == 0: continue

        if bin_conf[i] > bin_acc[i]:
            ax.bar(bin_centers[i], bin_conf[i], width=width, edgecolor="red", color="lightcoral", alpha=0.3, hatch="//", label="Confidence" if not labeled_higher_conf else "")
            ax.bar(bin_centers[i], bin_acc[i], width=width, edgecolor="black", color="royalblue", alpha=1.0, label="Accuracy" if not labeled_lower_acc else "")
            labeled_higher_conf = True
            labeled_lower_acc   = True
        
        else:
            ax.bar(bin_centers[i], bin_acc[i], width=width, edgecolor="black", color="royalblue", alpha=0.3, hatch="//", label="Accuracy" if not labeled_higher_acc else "")
            ax.bar(bin_centers[i], bin_conf[i], width=width, edgecolor="black", color="lightcoral", alpha=1.0, label="Confidence" if not labeled_lower_conf else "")
            labeled_higher_acc = True
            labeled_lower_conf = True

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Reliability Diagram", fontsize=14)

    ax.text(
        0.65,
        0.05,
        f"ECE = {ece * 100:.2f}%",
        fontsize=12,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
    )

    plt.grid(linestyle="--", alpha=0.3)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

def test():
    np.random.seed(42)
    n_samples = 1000

    y_prob = np.random.beta(a=5, b=2, size=n_samples)
    y_pred = np.random.randint(0, 2, size=n_samples)
    
    actual_accuracy = y_prob * 1.25
    is_correct = np.random.rand(n_samples) < actual_accuracy
    y_true = np.where(is_correct, y_pred, 1 - y_pred)

    print("Start plotting reliability diagram...")
    ece_value = plot_reliability_diagram(y_true, y_pred, y_prob)
    print(f"Calculated ece value: {ece_value:.4f}")


if __name__ == "__main__":
    test()
