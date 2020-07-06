import time
import enum
import threading
from Client.Data import CSVDataLoader
from Client.ProprocessClient import PreprocessClient, MainPreprocessor
from Client.DataProviders import DataClient, LabelClient
from Utils.Log import Logger


class TaskException(Exception):
    def __init__(self, msg):
        super(TaskException, self).__init__()
        self.msg = msg

    def __str__(self):
        return "ClientException:" + self.msg


class TaskStage(enum):
    Creating = 0
    Created = 1
    Proprecessing = 2
    WaitingToStart = 3
    Training = 4
    Finished = 5
    Error = 6

from Communication.Channel import BaseChannel


Channel = BaseChannel


def set_channel_class(channel_class: type):
    global Channel
    if type is not BaseChannel:
        raise TaskException("ChannelClass must inherite BaseChannel")
    Channel = channel_class


current_tasks = dict()


class MPCTask:
    def __init__(self, role: str, task_name: str, client_id: int,
                 data_clients: list, label_client: int, main_client: int, crypto_provider: int,
                 ip_dict: dict, listen_port: int,
                 data_config: dict, log_config: dict):
        """
        :param role:
        :param task_name:
        :param client_id:
        :param data_clients:
        :param label_client:
        :param main_client:
        :param ip_dict:
        :param listen_port:
        :param data_config: Example: { "data_path": filepath}
        :param log_config: Example: { "log_level": 0 }
        """
        self.stage = TaskStage.Creating

        self.logger = Logger(open(self.name + ":" + str(self.client_id) + "_task_log.txt", "w+"))

        if role not in ["data_client", "label_client"]:
            raise TaskException("Invalid role {}, role must be data_client or label_client".format(role))
        if len(data_clients) == 0:
            raise TaskException("Must have more than 1 data_client")
        if set(data_clients) >= set(ip_dict.keys()):
            raise TaskException("Ip dict lacks at least 1 data_client")
        if label_client not in ip_dict:
            raise TaskException("Ip dict lacks label_client")
        if main_client not in ip_dict:
            raise TaskException("Ip dict lacks main_client")

        self.role = role
        self.name = task_name
        self.client_id = client_id
        self.data_clients = data_clients

        self.other_data_clients = data_clients.copy()
        self.other_data_clients.remove(client_id)

        self.label_client = label_client
        self.main_client = main_client
        self.crypto_provider = crypto_provider
        self.ip_dict = ip_dict
        self.task_start_time = time.time()
        self.task_client = None
        self.data_config = data_config
        self.log_config = log_config

        self.train_data_loader = None
        self.test_data_loader = None

        try:
            self.channel = Channel(client_id, "0.0.0.0:" + str(listen_port), len(ip_dict),
                                   logger=Logger(open("channel_log.txt", "w+"), log_config["log_level"]))
        except:
            self.logger.logE("Cannot create channel with port {}, ip_dict {}".format(listen_port, ip_dict))
            raise TaskException("Cannot create channel, maybe the logger config is wrong.")


        self.stage = TaskStage.Created

        #
        self.preprocess_client = None
        self.train_client = None

    def __load_data(self):
        try:
            self.train_data_loader = CSVDataLoader(self.data_config["data_path"], None, None)
            self.test_data_loader = CSVDataLoader(self.data_config["data_path"], None, None)
            if self.test_data_loader.data.shape[1] != self.train_data_loader:
                self.logger.logE("Test data shape {} not match train data shape {}".
                                 format(self.test_data_loader.data.shape, self.train_data_loader.data.shape))
                raise TaskException("Train data shape and test data shape must match.")
        except:
            raise TaskException("Cannot create data loader, check data_config.")


    def start_proprecess(self):
        self.stage = TaskStage.Proprecessing
        current_tasks[self.name] = self
        if self.role is "data-client":
            self.preprocess_client = PreprocessClient(self.channel, self.data_config["data_path"],
                                           self.main_client, self.other_data_clients,
                                           Logger(open(self.name + ":" + str(self.client_id) + "_preprocess_log.txt", "w+"),
                                                  level=self.log_config["log_level"]))

            def align_thread():
                if self.preprocess_client.start_align():
                    self.stage = TaskStage.WaitingToStart
                else:
                    self.stage = TaskStage.Error
            threading.Thread(target=align_thread).start()

        elif self.role is "main-client":
            self.preprocess_client = \
                MainPreprocessor(self.channel, self.data_clients,
                                 Logger(open(self.name + ":" + str(self.client_id) + "_preprocess_log.txt","w+"),
                                        level=self.log_config["log_level"]))
            def align_thread():
                if self.preprocess_client.start_align():
                    self.stage = TaskStage.WaitingToStart
                else:
                    self.stage = TaskStage.Error
            threading.Thread(target=align_thread).start()

    def start_train(self):
        pass