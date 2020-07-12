import time
import tensorflow as tf
from Client.Learning.Metrics import AUC_KS
from Client.Data.DataLoader import CSVDataLoader
k = tf.keras

def test_credit_logistic(dims):
    train_data = CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000)), dims)
    train_data.set_random_seed({"seed": 8964})
    test_data = CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000, 50000)), dims)
    logistic_model = k.Sequential(k.layers.Dense(1, k.activations.sigmoid))
    logistic_model.compile(k.optimizers.SGD(0.1), k.losses.mean_squared_error)

    metrics = []
    start_time = time.time()
    for i in range(100000):
        if i % 1000 == 0:
            xys = test_data.get_batch(None)
            pred_ys = logistic_model.predict_on_batch(xys[:, :-1])
            metric = AUC_KS(xys[:, -1:], pred_ys)
            print("Train round {}, metric {}".format(i, metric))
            metrics.append([i, time.time() - start_time] + metric)
        else:
            xys = train_data.get_batch(32)
            loss = logistic_model.train_on_batch(xys[:, :-1], xys[:, -1:])
    return metrics

if __name__ == "__main__":
    import numpy as np
    metrics_full = test_credit_logistic(list(range(73)))
    metrics_0_29 = test_credit_logistic(list(range(30)) + [72])
    metrics_30_71 = test_credit_logistic(list(range(30, 73)))
    np.savetxt("full.csv", metrics_full, delimiter=",")
    np.savetxt("0-29.csv", metrics_0_29, delimiter=",")
    np.savetxt("30-71.csv", metrics_30_71, delimiter=",")