import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def AUC_KS(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)
    return [auc, ks]


metric_dict = {
        "auc": roc_auc_score,
        "auc_ks": AUC_KS
    }


def get_metric(metric_name: str):
    metric_name = metric_name.lower()
    return metric_dict[metric_name]

