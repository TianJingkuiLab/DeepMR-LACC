import numpy as np
from sksurv.metrics import concordance_index_censored


def compute_concordance_index(y_true, y_pred):                

    y_pred = y_pred.flatten()

    time_value = y_true[:, 0:1].flatten()                      
    event = y_true[:, 1:2].astype(bool).flatten()             

    result = concordance_index_censored(event, time_value,  y_pred)

    return result[0]


def compute_concordance_index_bootstrap_ci(y_true, y_pred, n_bootstrap=1000, confidence_level=0.95):

    c_indices = []

    for _ in range(n_bootstrap):

        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        bootstrap_y_true = y_true[indices]
        bootstrap_y_pred = y_pred[indices]

        if len(np.unique(bootstrap_y_true)) == 1:
            continue

        c_index = compute_concordance_index(bootstrap_y_true, bootstrap_y_pred)
        c_indices.append(c_index)

    c_indices.sort()

    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    lower_bound = np.percentile(c_indices, lower_percentile)
    upper_bound = np.percentile(c_indices, upper_percentile)

    return lower_bound, upper_bound
