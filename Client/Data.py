import numpy as np
class DataLoader:
    def sync_data(self, sync_info):
        raise NotImplementedError

    def get_batch(self, batch_size: int):
        raise NotImplementedError()


class RandomDataLoader(DataLoader):
    def __init__(self, dim):
        self.dim = dim

    def sync_data(self, sync_info):
        pass

    def get_batch(self, batch_size: int):
        return np.random.uniform(-1, 1, [batch_size, self.dim])


class CSVDataLoader(DataLoader):
    def __init__(self, csv_file_path, used_columns):
        csv_data = np.loadtxt(csv_file_path, delimiter=",")
        self.data = csv_data[:, used_columns]
        self.random_generator = None

    def sync_data(self, sync_info: dict):
        seed = sync_info["seed"]
        self.random_generator = np.random.default_rng(seed=seed)

    def get_batch(self, batch_size: int):
        indices = self.random_generator.choice(self.data.shape[0], batch_size)
        return self.data[indices]
