import numpy as np
import pandas as pd


class DataLoader:
    def set_random_seed(self, random_seed: int):
        raise NotImplementedError

    def get_batch(self, batch_size: int):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()


class CSVDataLoader(DataLoader):
    def __init__(self, csv_file_path, used_rows=None, used_columns=None):
        csv_data = pd.read_csv(csv_file_path, header=None).values
        if used_rows is None:
            used_rows = list(range(csv_data.shape[0]))
        if used_columns is None:
            used_columns = list(range(csv_data.shape[1]))
        self.data = csv_data[:, used_columns][used_rows]
        self.random_generator = None

    def set_random_seed(self, random_seed: int):
        seed = random_seed
        self.random_generator = np.random.default_rng(seed=seed)

    def get_batch(self, batch_size: int):
        if batch_size is None:
            return self.data
        indices = self.random_generator.choice(self.data.shape[0], batch_size)
        return self.data[indices]

    def shape(self):
        return self.data.shape[1:]