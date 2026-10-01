import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def compute_tsne(embeddings, n_components=2, perplexity=30, random_state=42):
    '''
    Args:
        embeddings(np.ndarray): shape (N, D)
    '''
    n_samples = embeddings.shape[0]
    safe_perplexity = max(5, min(perplexity, (n_samples - 1) // 3)) # Avoiding perplexity exceeding number of samples

    if safe_perplexity != perplexity:
        print(f"[t-SNE]\t [Warning]\t Clamped perplexity {perplexity} -> {safe_perplexity} for {n_samples} samples")

    # sklearn's TSNE(init="pca") hardcodes svd_solver="randomized" internally, which
    # triggers spurious numpy/Accelerate BLAS overflow warnings on macOS. Replicate
    # its PCA-init logic ourselves with the numerically-exact "full" solver instead.
    pca_init = PCA(n_components=n_components, svd_solver="full", random_state=random_state).fit_transform(embeddings)
    pca_init = (pca_init / np.std(pca_init[:, 0]) * 1e-4).astype(np.float32)

    tsne = TSNE(n_components=n_components, perplexity=safe_perplexity, random_state=random_state, init=pca_init)
    return tsne.fit_transform(embeddings)

TSNE_CLASS_COLORS = ["royalblue", "darkorange", "seagreen", "firebrick"]

def _plot_tsne_on_ax(ax, embeddings_2d, y_true, target_names, y_pred=None, title=None):
    '''
    Draw one t-SNE scatter panel onto `ax`. Shared by the single-plot and grid-plot entry points.

    Returns:
        int: number of misclassified points drawn (0 if y_pred is None)
    '''
    y_true = np.asarray(y_true)

    correct = None
    if y_pred is not None:
        correct = np.asarray(y_pred) == y_true

    n_wrong = 0
    for class_idx, class_name in enumerate(target_names):
        mask = y_true == class_idx
        color = TSNE_CLASS_COLORS[class_idx % len(TSNE_CLASS_COLORS)]

        if correct is None:
            ax.scatter(
                embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                s=20, alpha=0.7, color=color, label=f"{class_name} (n={mask.sum()})",
            )
            continue

        mask_correct = mask & correct
        mask_wrong = mask & ~correct
        n_wrong += mask_wrong.sum()

        ax.scatter(
            embeddings_2d[mask_correct, 0], embeddings_2d[mask_correct, 1],
            s=20, alpha=0.7, color=color, marker="o",
            label=f"{class_name} (n={mask.sum()}, wrong={mask_wrong.sum()})",
        )
        ax.scatter(
            embeddings_2d[mask_wrong, 0], embeddings_2d[mask_wrong, 1],
            s=40, alpha=0.9, color=color, marker="x", linewidths=1.5,
        )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    if title is not None:
        ax.set_title(title)
    ax.grid(alpha=0.3)

    return int(n_wrong)

def plot_tsne(embeddings_2d, y_true, target_names, y_pred=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))

    _plot_tsne_on_ax(ax, embeddings_2d, y_true, target_names, y_pred=y_pred)

    if y_pred is not None:
        ax.scatter([], [], marker="x", color="gray", linewidths=1.5, label="Misclassified")

    ax.set_title("TKR Abnormal Detection t-SNE Embedding")
    ax.legend(loc="best")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

def _balanced_grid_rows(n: int) -> list:
    '''
    Split n panels across as-square-as-possible rows, putting the remainder into
    the later rows (e.g. 5 -> [2, 3]) so every row is used and none is empty.
    '''
    n_cols = max(1, int(np.ceil(np.sqrt(n))))
    n_rows = int(np.ceil(n / n_cols))
    base, rem = divmod(n, n_rows)
    return [base + (1 if r >= n_rows - rem else 0) for r in range(n_rows)]

def plot_tsne_grid(embeddings_2d_list, y_true, target_names, y_pred_list=None, fold_labels=None, save_path=None):
    '''
    Draw one t-SNE panel per fold, wrapped into a balanced multi-row grid (e.g. 5 folds ->
    row of 2 + row of 3, each centered), saved as a single combined figure.

    Args:
        embeddings_2d_list(list[np.ndarray]): one (N, 2) array per fold
        y_true(list): shared true labels (same test set across folds)
        target_names(list): class name for each label index
        y_pred_list(list, optional): one predicted-label array per fold; when given, that fold's
            misclassified points are marked with "x"
        fold_labels(list[str], optional): panel titles, defaults to "Fold 0", "Fold 1", ...
    '''
    n_folds = len(embeddings_2d_list)
    if fold_labels is None:
        fold_labels = [f"Fold {i}" for i in range(n_folds)]
    if y_pred_list is None:
        y_pred_list = [None] * n_folds

    row_counts = _balanced_grid_rows(n_folds)
    n_rows = len(row_counts)
    n_cols = max(row_counts)

    fig = plt.figure(figsize=(6 * n_cols, 6 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols * 2)

    axes = []
    for row, count in enumerate(row_counts):
        offset = n_cols - count
        for col in range(count):
            start = offset + col * 2
            axes.append(fig.add_subplot(gs[row, start:start + 2]))

    for i in range(n_folds):
        n_wrong = _plot_tsne_on_ax(axes[i], embeddings_2d_list[i], y_true, target_names, y_pred=y_pred_list[i])
        title = fold_labels[i]
        axes[i].set_title(title)

    handles = [
        plt.Line2D([], [], marker="o", color=TSNE_CLASS_COLORS[idx % len(TSNE_CLASS_COLORS)], linestyle="", label=name)
        for idx, name in enumerate(target_names)
    ]
    if any(yp is not None for yp in y_pred_list):
        handles.append(plt.Line2D([], [], marker="x", color="gray", linestyle="", label="Misclassified"))

    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02), fontsize=20)
    fig.suptitle("TKR Abnormal Detection t-SNE Embedding Across Folds", fontsize=20)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

def tsne_analysis(embeddings, y_true, target_names, y_pred=None, save_path=None, **tsne_kwargs):
    '''
    Args:
        embeddings(np.ndarray): shape (N, D) pre-classifier feature vectors
        y_true(list): labels (0 is normal | 1 is abnormal)
        target_names(list): class name for each label index
        y_pred(list, optional): predicted labels; when given, misclassified points are marked with "x"
    '''
    embeddings = np.asarray(embeddings)
    print(f"[t-SNE]\t Reducing {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim) to 2D...")

    embeddings_2d = compute_tsne(embeddings, **tsne_kwargs)
    plot_tsne(embeddings_2d, y_true, target_names, y_pred=y_pred, save_path=save_path)

    print(f"[t-SNE]\t Done. {'Saved to ' + save_path if save_path else 'Displayed.'}")

    return embeddings_2d

def build_perplexity_path(save_path: str, perplexity: int) -> str:
    '''
    Insert the perplexity value into a filename before its extension.
    e.g. "tsne.png", 30 -> "tsne_30.png"
    '''
    root, ext = os.path.splitext(save_path)
    return f"{root}_{perplexity}{ext}"

def tsne_analysis_sweep(embeddings, y_true, target_names, y_pred=None, save_path=None, perplexities=(30,), **tsne_kwargs):
    '''
    Run tsne_analysis once per perplexity value, saving one plot per value.

    Args:
        save_path(str): base path, e.g. "tsne.png" -> "tsne_30.png", "tsne_50.png", ...
        perplexities(list[int]): perplexity values to try
        y_pred(list, optional): predicted labels; when given, misclassified points are marked with "x"
    Returns:
        dict[int, np.ndarray]: perplexity -> 2D embedding
    '''
    results = {}

    for perplexity in perplexities:
        perplexity_path = build_perplexity_path(save_path, perplexity) if save_path is not None else None
        print(f"[t-SNE]\t Running sweep for perplexity={perplexity}")

        results[perplexity] = tsne_analysis(
            embeddings, y_true, target_names, y_pred=y_pred, save_path=perplexity_path, perplexity=perplexity, **tsne_kwargs
        )

    return results

def tsne_analysis_grid(embeddings_list, y_true, target_names, y_pred_list=None, fold_labels=None, save_path=None, **tsne_kwargs):
    '''
    Run t-SNE independently per fold (each fold has its own raw embedding space) and combine
    the resulting panels into a single saved figure.

    Args:
        embeddings_list(list[np.ndarray]): one (N, D) embedding array per fold, same N/order across folds
        y_true(list): shared true labels (same test set across folds)
        target_names(list): class name for each label index
        y_pred_list(list, optional): one predicted-label array per fold
        fold_labels(list[str], optional): panel titles, defaults to "Fold 0", "Fold 1", ...
    Returns:
        list[np.ndarray]: one 2D embedding per fold
    '''
    embeddings_2d_list = []
    for i, embeddings in enumerate(embeddings_list):
        embeddings = np.asarray(embeddings)
        print(f"[t-SNE]\t Fold {i}: reducing {embeddings.shape[0]} embeddings ({embeddings.shape[1]}-dim) to 2D...")
        embeddings_2d_list.append(compute_tsne(embeddings, **tsne_kwargs))

    plot_tsne_grid(embeddings_2d_list, y_true, target_names, y_pred_list=y_pred_list, fold_labels=fold_labels, save_path=save_path)

    print(f"[t-SNE]\t Done. {'Saved to ' + save_path if save_path else 'Displayed.'}")

    return embeddings_2d_list

def tsne_analysis_grid_sweep(embeddings_list, y_true, target_names, y_pred_list=None, fold_labels=None, save_path=None, perplexities=(30,), **tsne_kwargs):
    '''
    Run tsne_analysis_grid once per perplexity value, saving one combined grid image per value.

    Args:
        save_path(str): base path, e.g. "tsne_grid.png" -> "tsne_grid_30.png", "tsne_grid_50.png", ...
        perplexities(list[int]): perplexity values to try
    Returns:
        dict[int, list[np.ndarray]]: perplexity -> list of per-fold 2D embeddings
    '''
    results = {}

    for perplexity in perplexities:
        perplexity_path = build_perplexity_path(save_path, perplexity) if save_path is not None else None
        print(f"[t-SNE]\t Running grid sweep for perplexity={perplexity}")

        results[perplexity] = tsne_analysis_grid(
            embeddings_list, y_true, target_names, y_pred_list=y_pred_list,
            fold_labels=fold_labels, save_path=perplexity_path, perplexity=perplexity, **tsne_kwargs
        )

    return results

def test():
    np.random.seed(42)
    normal_size   = 178
    abnormal_size = 35

    normal_embeds   = np.random.normal(loc=0.0, scale=1.0, size=(normal_size, 64))
    abnormal_embeds = np.random.normal(loc=3.0, scale=1.0, size=(abnormal_size, 64))
    embeddings = np.concatenate([normal_embeds, abnormal_embeds], axis=0)

    y_true = np.concatenate([np.zeros(normal_size), np.ones(abnormal_size)])

    indices = np.arange(len(y_true))
    np.random.shuffle(indices)
    embeddings = embeddings[indices]
    y_true = y_true[indices]

    # Fabricate predictions with a handful of flipped labels to exercise the misclassification markers
    y_pred = y_true.copy()
    flip_indices = np.random.choice(len(y_pred), size=15, replace=False)
    y_pred[flip_indices] = 1 - y_pred[flip_indices]

    tsne_analysis_sweep(embeddings, y_true, target_names=["Normal", "Abnormal"], y_pred=y_pred, perplexities=[5, 30, 50])

    test_grid(embeddings, y_true)

def test_grid(embeddings, y_true, n_folds=5):
    '''
    Fabricate slightly different embeddings/predictions per fold to exercise the combined
    grid-plotting path (plot_tsne_grid / tsne_analysis_grid_sweep).
    '''
    embeddings_list = []
    y_pred_list = []

    for fold in range(n_folds):
        rng = np.random.RandomState(fold)
        fold_embeddings = embeddings + rng.normal(loc=0.0, scale=0.5, size=embeddings.shape)
        embeddings_list.append(fold_embeddings)

        fold_y_pred = y_true.copy()
        flip_indices = rng.choice(len(fold_y_pred), size=10 + fold, replace=False)
        fold_y_pred[flip_indices] = 1 - fold_y_pred[flip_indices]
        y_pred_list.append(fold_y_pred)

    tsne_analysis_grid_sweep(
        embeddings_list, y_true, target_names=["Normal", "Abnormal"],
        y_pred_list=y_pred_list, perplexities=[30],
    )

if __name__ == "__main__":
    test()
