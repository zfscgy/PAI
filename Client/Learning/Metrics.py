import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, f1_score, recall_score, log_loss


def AUC_KS(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)
    return [auc, ks]


def metrics_pack1(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return: ACC, F1, Precision, Recall, logLoss, AUC, KS
    """
    return [accuracy_score(y_true, np.round(y_pred)), f1_score(y_true, np.round(y_pred)),
            precision_score(y_true, np.round(y_pred)), recall_score(y_true, np.round(y_pred)),
            log_loss(y_true, y_pred)] + AUC_KS(y_true, y_pred)


metric_dict = {
        "auc": roc_auc_score,
        "auc_ks": AUC_KS,
        "metrics_pack1": metrics_pack1
    }


def get_metric(metric_name: str):
    metric_name = metric_name.lower()
    return metric_dict[metric_name]

