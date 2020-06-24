import tensorflow as tf
from sklearn.metrics import roc_auc_score

from Client.Data import CSVDataLoader
k = tf.keras

def test_credit_logistic():
    train_data = CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000)), list(range(73)))
    train_data.sync_data({"seed": 8964})
    test_data = CSVDataLoader("Test/TestDataset/credit_default.csv", list(range(40000, 50000)), list(range(73)))
    logistic_model = k.Sequential(k.layers.Dense(1, k.activations.sigmoid))
    logistic_model.compile(k.optimizers.SGD(0.1), k.losses.mean_squared_error)
    for i in range(50000):
        if i % 100 == 0:
            xys = test_data.get_batch(None)
            pred_ys = logistic_model.predict_on_batch(xys[:, :-1])
            auc = roc_auc_score( xys[:, -1:], pred_ys)
            print("Train round %d, auc %.4f" % (i, auc))
        else:
            xys = train_data.get_batch(32)
            loss = logistic_model.train_on_batch(xys[:, :-1], xys[:, -1:])

if __name__ == "__main__":
    test_credit_logistic()