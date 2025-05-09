import logging
from typing import *
from sklearn.metrics import precision_recall_curve, f1_score, auc
import numpy as np
def update_dict_nonnull(d: Dict[str, Any], vals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a dictionary with values from another dictionary.
    >>> update_dict_nonnull({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    {'a': 1, 'b': 3, 'c': 4}
    """
    for k, v in vals.items():
        if k in d:
            if d[k] != v and v is not None:
                logging.info(f"Replacing key {k} original value {d[k]} with {v}")
                d[k] = v
        else:
            d[k] = v
    return d
def calculate_aupr_fmax(labels,preds):
    auprs = []
    fmax_scores = []
    for i in range(labels.shape[1]):
        precision, recall, _ = precision_recall_curve(labels[:, i], preds[:, i])
        pr_auc = auc(recall, precision)
        auprs.append(pr_auc)
        f1_scores = 2 * (precision * recall) / (precision + recall + np.finfo(float).eps)
        fmax = np.max(f1_scores)
        fmax_scores.append(fmax)

    mean_aupr = np.mean(auprs)
    mean_fmax = np.mean(fmax_scores)
    
    return mean_aupr,mean_fmax