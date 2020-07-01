import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def AUC_KS(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)
    return [auc, ks]