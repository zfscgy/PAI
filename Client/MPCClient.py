import os
from enum import Enum

from Client.Client import BaseClient
from Client.Data.DataLoader import DataLoader
from Communication.Channel import BaseChannel
from Utils.Log import Logger


class MPCClientException(Exception):
    def __init__(self, client_class: type, message:str):
        self.client_class = client_class
        self.message = message

    def __str__(self):
        return "MPCClientException in {}, message {}".format(self.client_class, self.message)


class MPCClientParas:
    def __init__(self, feature_client_ids: list, label_client_id: int, main_client_id: int, crypto_producer_id: int,
                 other_clients: dict=None):
        self.feature_client_ids = feature_client_ids
        self.label_client_id = label_client_id
        self.main_client_id = main_client_id
        self.crypto_producer_id = crypto_producer_id
        self.other_clients = other_clients


class ClientMode(Enum):
    Train = 0
    Test = 1
    Predict = 2
    Any = 3


class MPCClient(BaseClient):
    def __init__(self, channel: BaseChannel, logger: Logger, mpc_paras: MPCClientParas):
        super(MPCClient, self).__init__(channel, logger)
        self.feature_client_ids = mpc_paras.feature_client_ids
        self.label_client_id = mpc_paras.label_client_id
        self.main_client_id = mpc_paras.main_client_id
        self.crypto_producer_id = mpc_paras.crypto_producer_id
        self.mpc_mode = ClientMode.Train


class PreprocessClient(MPCClient):
    def __init__(self, channel: BaseChannel, logger: Logger, mpc_paras: MPCClientParas,
                 source_data_file: str, out_data_dir: str):
        """
        :param channel:
        :param logger:
        :param mpc_paras:
        :param source_data_file:
        :param out_data_dir:
        """
        super(PreprocessClient, self).__init__(channel, logger, mpc_paras)
        self.other_data_client_ids = mpc_paras.feature_client_ids.copy()
        self.other_data_client_ids.remove(self.client_id)
        if os.path.isfile(source_data_file):
            self.source_data_file = source_data_file
        else:
            raise MPCClientException(PreprocessClient, "source_data_file not exists")
        if os.path.isdir(out_data_dir):
            self.out_data_dir = out_data_dir
        else:
            raise MPCClientException(PreprocessClient, "out_data_dir is not a directory")

    def start_preprocess(self):
        raise NotImplementedError()


class DataClient(MPCClient):
    def __init__(self, channel: BaseChannel, logger: Logger, mpc_paras: MPCClientParas,
                 train_data_loader: DataLoader, test_data_loader: DataLoader):
        """
        :param channel:
        :param logger:
        :param mpc_paras:
        :param train_data_loader:
        :param test_data_loader:
        """
        super(DataClient, self).__init__(channel, logger, mpc_paras)
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.data_shape = train_data_loader.shape()

    def start_train(self) -> bool:
        raise NotImplementedError()

    def start_service(self) -> bool:
        raise NotImplementedError()


class MainClient(MPCClient):
    def __init__(self, channel: BaseChannel, logger: Logger,
                 mpc_paras: MPCClientParas):
        super(MainClient, self).__init__(channel, logger, mpc_paras)

    def start_train(self) -> bool:
        raise NotImplementedError()

    def start_service(self) -> bool:
        raise NotImplementedError()