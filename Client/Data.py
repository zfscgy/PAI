import numpy as np
class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sync_data(self, sync_info):
        raise NotImplementedError

    def get_train_batch(self, batch_size=None):
        raise NotImplementedError()

    def get_test_batch(self, batch_size=None):
        raise NotImplementedError()


class RandomDataLoader(DataLoader):
    def __init__(self, dim, batch_size):
        self.dim = dim
        self.batch_size = batch_size

    def sync_data(self, sync_info):
        pass

    def get_train_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return np.random.uniform(-1, 1, [batch_size, self.dim])

    def get_test_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return np.random.uniform(-1, 1, [batch_size, self.dim])