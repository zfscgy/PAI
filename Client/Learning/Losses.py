import numpy as np


class LossException(Exception):
    def __init__(self, msg):
        super(LossException, self).__init__()
        self.msg = msg

    def __str__(self):
        return "LossException:" + self.msg


class LossFunc:
    def forward(self, ys, pred_ys):
        raise NotImplementedError()

    def backward(self):
        raise NotImplementedError()


class MSELoss(LossFunc):
    def __init__(self):
        self.ys = None
        self.pred_ys = None

    def forward(self, ys, pred_ys):
        return np.mean(np.square(ys - pred_ys))

    def backward(self):
        if self.ys is None:
            raise LossException("Cannot backward before forward")
        grad = 2 * (self.pred_ys - self.ys)
        self.ys = None
        self.pred_ys = None
        return grad