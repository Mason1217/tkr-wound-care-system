import numpy as np

def bootstrap_ci(y_true, y_pred, metric_func, n_iter=2000, ci=95, **metric_kwargs):
    '''
    Args:
        y_true(list): true labels
        y_pred(list): model predictions
        metric_func(callable): evaluation function in sklearn.metrics (e.g. recall_score)
        n_iter(int)
        ci(int): percentage of confidence interval
        **metric_kwargs: kwargs for metric function (e.g. pos_lable=1)

    Returns:
        mean_score, lower_bound, upper_bound
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_size = len(y_true)
    stats  = []

    for i in range(n_iter):
        indices = np.random.choice(n_size, size=n_size, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        if len(np.unique(y_true_boot)) < 2: continue

        score = metric_func(y_true_boot, y_pred_boot, **metric_kwargs)
        stats.append(score)

    alpha   = (100 - ci) / 2
    lower_p = alpha
    upper_p = 100 - alpha

    lower_bound = np.percentile(stats, lower_p)
    upper_bound = np.percentile(stats, upper_p)
    mean_score  = np.mean(stats)

    return mean_score, lower_bound, upper_bound
