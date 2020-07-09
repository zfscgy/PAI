import time
import enum
import threading
import os
from Client.Data import CSVDataLoader
from Client.ProprocessClient import PreprocessClient, MainPreprocessor
from Client.SMCProvider import TripletsProvider
from Client.DataProviders import DataClient, LabelClient
from Client.ComputationProviders import MainTFClient
from Client.Learning.Metrics import get_metric
from Client.Learning.Losses import get_loss
from Utils.Log import Logger
import Server.TaskConfig as Config

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


Current_tasks = dict()


class MPCTask:
    def __init__(self, role: str, task_name: str, client_id: int,
                 data_clients: list, label_client: int, main_client: int, crypto_provider: int,
                 ip_dict: dict, listen_port: int,
                 # data-clients, label-client
                 data_config: dict,
                 # label-client
                 learn_config: dict,
                 # main-client
                 train_config: dict,
                 # 本地设定
                 log_config: dict):
        """
        :param role: one of "data-client", "label-client", "main-client"
        :param task_name:

        :param client_id:
        :param data_clients:
        :param label_client:
        :param main_client:
        :param crypto_provider:

        :param ip_dict:
        :param listen_port:

        :param data_config:
            Example: { "data_path": filepath}
        :param learn_config:
            Example: { "loss": "mse", "metrics": "auc_ks" }
        :param train_config:
            Only main client need this
            Example: { "client_dims": {2: 100, 3: 50}, "layers": [10, 1],
                       "batch_size": 32, "test_per_batch":1001, "learning_rate": 0.1, "max_iter": 10011 }
        :param log_config: Example: { "log_level": 0 }
        """

        self.stage = TaskStage.Creating

        self.logger = Logger(open(Config.TaskRootPath + self.name + ":" + str(self.client_id) + "_task_log.txt", "w+"))

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
        self.learn_config = learn_config
        self.train_config = train_config
        self.log_config = log_config

        self.train_data_loader = None
        self.test_data_loader = None

        try:
            self.channel = Channel(client_id, "0.0.0.0:" + str(listen_port), len(ip_dict),
                                   logger=Logger(open("channel_log.txt", "w+"), log_config["log_level"]))
        except:
            self.logger.logE("Cannot create channel with port {}, ip_dict {}".format(listen_port, ip_dict))
            raise TaskException("Cannot create channel, maybe the logger config is wrong.")

        try:
            os.mkdir(Config.TaskRootPath + task_name)
            os.mkdir(Config.TaskRootPath + task_name + "/log")
        except:
            self.logger.logE("Cannot create task directory with root path {} and task_name {}".
                             format(Config.TaskRootPath, task_name))
            raise TaskException("Cannot create task directory")

        self.preprocess_client = None
        self.train_client = None
        Current_tasks[self.name] = self
        self.stage = TaskStage.Created

        self.current_task_path = Config.TaskRootPath + task_name + "/"

    def __load_data(self):
        try:
            self.train_data_loader = CSVDataLoader(Config.DataRootPath + self.data_config["data_path"][:-4] + "_train.csv", None, None)
            self.test_data_loader = CSVDataLoader(Config.DataRootPath + self.data_config["data_path"][:-4] + "_test.csv", None, None)
            if self.test_data_loader.data.shape[1] != self.train_data_loader:
                self.logger.logE("Test data shape {} not match train data shape {}".
                                 format(self.test_data_loader.data.shape, self.train_data_loader.data.shape))
                raise TaskException("Train data shape and test data shape must match.")
        except:
            raise TaskException("Cannot create data loader, check data_config.")

    def start_preprocess(self, join=False):
        self.stage = TaskStage.Proprecessing
        if self.role is "data-client" or self.role is "label-client":
            self.preprocess_client = \
                PreprocessClient(self.channel, Config.DataRootPath + self.data_config["data_path"],
                                 self.main_client, self.other_data_clients,
                                 Logger(open(self.current_task_path + "log/" + self.name + ":" + str(self.client_id) + "_preprocess_log.txt", "w+"),
                                        level=self.log_config["log_level"]))

            def align_thread():
                if self.preprocess_client.start_align():
                    self.stage = TaskStage.WaitingToStart
                else:
                    self.stage = TaskStage.Error

        elif self.role is "main-client":
            self.preprocess_client = \
                MainPreprocessor(self.channel, self.data_clients,
                                 Logger(open(self.current_task_path + "log/" + self.name + "-" + str(self.client_id) + "_preprocess_log.txt", "w+"),
                                        level=self.log_config["log_level"]))
            def align_thread():
                if self.preprocess_client.start_align():
                    self.stage = TaskStage.WaitingToStart
                else:
                    self.stage = TaskStage.Error

        elif self.role is "crypto-producer":
            self.train_client = TripletsProvider(self.channel, self.data_clients,
                                                 Logger(open(self.current_task_path + "log/" + self.name + "-" + str(self.client_id) + "_triplets_log.txt", "w+"),
                                                        level=self.log_config["log_level"]))
            def align_thread():
                self.train_client.start_listening()

        else:
            # It should never be reached
            return False

        thread = threading.Thread(target=align_thread)
        thread.start()
        if join:
            thread.join()


    def start_train(self, join=False):
        if self.role is "data-client" or self.role is "label-client":
            if self.role is "data-client":
                self.train_client = DataClient(self.channel, self.train_data_loader, self.test_data_loader,
                                               self.main_client, self.crypto_provider, self.other_data_clients,
                                               Logger(open(self.current_task_path + "log/" + self.name + "-" + str(self.client_id) + "_train_log.txt", "w+"),
                                                      level=self.log_config["log_level"]))
            else:
                self.train_client = LabelClient(self.channel, self.train_data_loader, self.test_data_loader,
                                                self.main_client,
                                                get_loss(self.train_config["loss"]),
                                                get_metric(self.train_config["metrics"]),
                                                Logger(open(self.current_task_path + "log/" + self.name + "-" + str(self.client_id) + "_train_log.txt", "w+"),
                                                       level=self.log_config["log_level"]))
            def train_thread():
                if self.train_client.start_train():
                    self.stage = TaskStage.Finished
                else:
                    self.stage = TaskStage.Error
            threading.Thread(target=train_thread).start()

        elif self.role is "main-client":
            self.train_client = MainTFClient(self.channel, self.data_clients, self.label_client,
                                             Logger(open(self.current_task_path + "log/" + self.name + "-" + str(self.client_id) + "_train_log.txt", "w+")))
            def train_thread():
                if self.train_client.start_train():
                    self.stage = TaskStage.Finished
                else:
                    self.stage = TaskStage.Error
        elif self.role is "crypto-producer":
            return True
        else:
            return False

        thread = threading.Thread(target=train_thread)
        thread.start()
        if join:
            thread.join()


def create_task(**kwargs):
    try:
        task = MPCTask(**kwargs)
    except:
        return None
    return task
