import time
import enum
import threading
import os
import pickle
import json
import re
from Client.Data import CSVDataLoader
from Client.ProprocessClient import PreprocessClient, MainPreprocessor
from Client.SMCProvider import TripletsProvider
from Client.DataProviders import DataClient, LabelClient

from Client.Learning.Metrics import get_metric
from Client.Learning.Losses import get_loss
from Utils.Log import Logger
import Server.TaskConfig as Config

from Communication.protobuf import message_pb2_grpc, message_pb2

class TaskException(Exception):
    def __init__(self, msg):
        super(TaskException, self).__init__()
        self.msg = msg

    def __str__(self):
        return "ClientException:" + self.msg


class TaskStage(enum.Enum):
    Creating = 0
    Created = 1
    Proprecessing = 2
    WaitingToStart = 3
    Training = 4
    Finished = 5
    Error = 6

from Communication.Channel import BaseChannel
from Communication.RPCComm import Peer as GRPCChannel


Channel = GRPCChannel


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
                 # 本地设定
                 log_config: dict,
                 # data_clients, label_client
                 data_config: dict=None,
                 # label_client
                 learn_config: dict=None,
                 # main_client
                 train_config: dict=None,
):
        """
        :param role: one of "data_client", "label_client", "main_client"
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
        self.current_task_path = Config.TaskRootPath + task_name + "-%d" % client_id + "/"

        self.stage = TaskStage.Creating

        self.logger = Logger(open(self.current_task_path + "task_log.txt", "w+"))

        # Since ip_dict maybe serialized from json. The key of ip_dict has to be int.
        new_ip_dict = {}
        for id_str in ip_dict:
            addr = ip_dict[id_str]
            new_ip_dict[int(id_str)] = addr
        ip_dict = new_ip_dict

        if role not in ["data_client", "label_client", "main_client", "crypto_producer"]:
            msg = "Invalid role {}, role must be data_client or label_client.".format(role)
            self.logger.logE(msg)
        if len(data_clients) == 0:
            self.logger.logE("Must have more than 1 data_client")
        if set(data_clients) >= set(ip_dict.keys()):
            self.logger.logE("Ip dict lacks at least 1 data_client. Ip_dict:\n" + str(ip_dict))
        if label_client not in ip_dict:
            self.logger.logE("Ip dict lacks label_client. Ip_dict:\n" + str(ip_dict))
        if main_client not in ip_dict:
            self.logger.logE("Ip dict lacks main_client. Ip_dict:\n" + str(ip_dict))

        self.role = role
        self.name = task_name
        self.client_id = client_id
        self.data_clients = data_clients

        self.other_data_clients = data_clients.copy()
        if role is "data_client":
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
            os.mkdir(Config.TaskRootPath + task_name + "-%d" % client_id + "/log")
        except:
            self.logger.logE("Cannot create task log directory with root path {} and task_name {}".
                             format(Config.TaskRootPath, task_name))

        try:
            self.channel = Channel(client_id, "0.0.0.0:" + str(listen_port), len(ip_dict), ip_dict, time_out=30,
                                   logger=Logger(open(self.current_task_path + "log/channel_log.txt", "w+"), level=log_config["log_level"]))
        except:
            self.logger.logE("Cannot create channel with port {}, ip_dict {}".format(listen_port, ip_dict))

        if add_query_service_to_computation_gprc_server(self):
            self.logger.log("Attach query service to computation grpc server. Available to query stask status")
        else:
            self.logger.logW("Cannot attach query service since compuation server is not grpc server.")
        self.preprocess_client = None
        self.train_client = None
        Current_tasks[self.name] = self
        self.stage = TaskStage.Created


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
        if self.role is "data_client" or self.role is "label_client":
            self.preprocess_client = \
                PreprocessClient(self.channel, Config.DataRootPath + self.data_config["data_path"],
                                 self.main_client, self.other_data_clients,
                                 Logger(open(self.current_task_path + "log/preprocess_log.txt", "w+"),
                                        level=self.log_config["log_level"]))

            def align_thread():
                if self.preprocess_client.start_align():
                    self.stage = TaskStage.WaitingToStart
                else:
                    self.stage = TaskStage.Error

        elif self.role is "main_client":
            self.preprocess_client = \
                MainPreprocessor(self.channel, self.data_clients,
                                 Logger(open(self.current_task_path + "log/preprocess_log.txt", "w+"),
                                        level=self.log_config["log_level"]))
            def align_thread():
                if self.preprocess_client.start_align():
                    self.stage = TaskStage.WaitingToStart
                else:
                    self.stage = TaskStage.Error

        elif self.role is "crypto_producer":
            self.train_client = TripletsProvider(self.channel, self.data_clients,
                                                 Logger(open(self.current_task_path + "log/triplets_log.txt", "w+"),
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
        if self.role is "data_client" or self.role is "label_client":
            try:
                self.__load_data()
            except:
                self.logger.logE("Load data failed. Stop training")
                return False
            if self.role is "data_client":
                self.train_client = DataClient(self.channel, self.train_data_loader, self.test_data_loader,
                                               self.main_client, self.crypto_provider, self.other_data_clients,
                                               Logger(open(self.current_task_path + "log/train_log.txt", "w+"),
                                                      level=self.log_config["log_level"]))
            else:
                self.train_client = LabelClient(self.channel, self.train_data_loader, self.test_data_loader,
                                                self.main_client,
                                                get_loss(self.learn_config["loss"]),
                                                get_metric(self.learn_config["metrics"]),
                                                Logger(open(self.current_task_path + "log/train_log.txt", "w+"),
                                                       level=self.log_config["log_level"]))
            def train_thread():
                if self.train_client.start_train():
                    self.stage = TaskStage.Finished
                else:
                    self.stage = TaskStage.Error
            threading.Thread(target=train_thread).start()

        elif self.role is "main_client":
            from Client.ComputationProviders import MainTFClient
            self.train_client = MainTFClient(self.channel, self.data_clients, self.label_client, self.train_config,
                                             Logger(open(self.current_task_path + "log/train_log.txt", "w+")))
            time.sleep(70)  # Waiting for clients to load data
            def train_thread():
                if self.train_client.start_train():
                    self.stage = TaskStage.Finished
                else:
                    self.stage = TaskStage.Error
        elif self.role is "crypto_producer":
            return True
        else:
            return False

        thread = threading.Thread(target=train_thread)
        thread.start()
        if join:
            thread.join()


    def start_all(self, join=False):
        def task_thread():
            self.start_preprocess(join=True)
            self.start_train(join=True)
        thread = threading.Thread(target=task_thread)
        thread.start()
        if join:
            thread.join()


def create_task_pyscript(**kwargs):

    os.mkdir(Config.TaskRootPath + kwargs["task_name"] + "-%d" % (kwargs["client_id"]))
    task_script_string = \
        "import sys, os\n" +\
        "sys.path.append('{}')\n".format(re.escape(os.getcwd())) +\
        "os.chdir('{}')\n".format(re.escape(os.getcwd())) +\
        "from Server.TaskControl import MPCTask\n" +\
        "task_config = {}\n".format(json.dumps(kwargs, indent=4)) +\
        "task = MPCTask(**task_config)\n" +\
        "task.start_all()\n"
    pyscripte_file = open(Config.TaskRootPath + kwargs["task_name"] + "-%d" % kwargs["client_id"] + "/task.py", "w+")
    pyscripte_file.write(task_script_string)
    pyscripte_file.close()


class QueryMPCTaskServicer(message_pb2_grpc.QueryMPCTaskServicer):
    def __init__(self, task: MPCTask):
        self.task = task
        self.query_dict = {
            "QueryStage": self._query_stage,
            "StartTask": self._start_task,
            "QueryNRounds": self._query_n_rounds,
            "QueryStats": self._query_stats
        }

    def _start_task(self):
        if self.task.stage is not TaskStage.Created:
            return "The task is already started before", None
        self.task.start_all()
        return "ok", None

    def _query_stage(self):
        return self.task.stage.name

    def _query_n_rounds(self):
        if self.task.role == "crypto_producer":
            # This code shall never reached
            return "crypto_producer do not have rounds", None
        return "ok", self.task.train_client.n_rounds

    def _query_stats(self):
        if type(self.task.train_client) is not LabelClient:
            return "Only label client have stats", None
        assert isinstance(self.task.train_client, LabelClient)
        return "ok", self.task.train_client.metrics_record[-5:]

    def QueryTask(self, request: message_pb2.TaskQuery, context):
        def encode(status: str, obj=None):
            return message_pb2.TaskResponse(status=status, python_bytes=pickle.dumps(obj))
        if request.query_string not in self.query_dict:
            return encode("Query url not exist")
        status, resp = self.query_dict[request.query_string]()
        return encode(status, resp)


def add_query_service_to_computation_gprc_server(task: MPCTask):
    from Communication.RPCComm import Peer
    peer = task.channel
    if not isinstance(peer, Peer):
        return False
    message_pb2_grpc.add_QueryMPCTaskServicer_to_server(QueryMPCTaskServicer(task), peer.server.server)
    return True
