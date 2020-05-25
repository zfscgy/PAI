import numpy as np


def onehot_accuracy(ys, pred_ys):
    acc = np.mean(np.argmax(ys, 1) == np.argmax(pred_ys, 1))
    return acc