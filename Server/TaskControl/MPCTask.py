import time
import enum
import threading
import os
import pickle
import json
import re
from Client.MPCClient import MPCClientParas
from Client.Data.DataLoader import CSVDataLoader
from Client.Preprocess.PreprocessClient import PreprocessClient, MainPreprocessor
from Client.MPCProviders.SMCProvider import TripletsProvider
from Client.SharedNN.DataProviders import FeatureClient, LabelClient
from Client.Learning.Metrics import get_metric
from Client.Learning.Losses import get_loss
from Utils.Log import Logger
import Server.TaskControl.TaskConfig as Config

from Communication.protobuf import message_pb2_grpc, message_pb2


class TaskException(Exception):
    def __init__(self, msg):
        super(TaskException, self).__init__()
        self.msg = msg

    def __str__(self):
        return "TaskException:" + self.msg


class MPCRoles:
    roles = {"feature_provider", "label_provider", "main_provider", "crypto_provider"}
    FeatureProvider = "feature_provider"
    LabelProvider = "label_provider"
    MainProvider = "main_provider"
    CryptoProvider = "crypto_provider"

class TaskStage(enum.Enum):
    Creating = 0
    Created = 1
    Proprecessing = 2
    LoadingData = 3
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



class MPCTask:
    def __init__(self, role: str, task_name: str,
                 client_id: int, mpc_paras: MPCClientParas,
                 ip_dict: dict, listen_port: int,
                 log_level: int, configs: dict
):
        """
        :param role: one of "data_client", "label_client", "main_client"
        :param task_name:

        :param client_id:

        :param mpc_paras:
         - feature_clients:
         - label_client:
         - main_client:
         - crypto_provider:

        :param ip_dict:
        :param listen_port:

        :param config
        :param log_level:
        """
        if role not in MPCRoles.roles:
            self.logger.logE("Not ")
            self.stage = TaskStage.Error
        self.role = role

        self.current_task_path = Config.TaskRootPath + task_name + "-%d" % client_id + "/"
        self.stage = TaskStage.Creating
        self.log_level = log_level
        self.logger = Logger(open(self.current_task_path + "task_log.txt", "w+"), level=log_level)

        # Since ip_dict maybe serialized from json. The key of ip_dict has to be int.
        new_ip_dict = {}
        for id_str in ip_dict:
            addr = ip_dict[id_str]
            new_ip_dict[int(id_str)] = addr
        ip_dict = new_ip_dict
        self.ip_dict = ip_dict

        if isinstance(mpc_paras, dict):
            try:
                self.mpc_paras = MPCClientParas(**mpc_paras)
            except:
                self.logger.logE("mpc_paras not correct")
                self.stage = TaskStage.Error
                return
        elif not isinstance(mpc_paras, MPCClientParas):
            self.logger.logE("mpc_paras must be a dict or a MPCClientParas instance")
            self.stage = TaskStage.Error

        if role not in ["data_client", "label_client", "main_client", "crypto_producer"]:
            msg = "Invalid role {}, role must be data_client or label_client.".format(role)
            self.logger.logE(msg)
            self.stage = TaskStage.Error

        self.name = task_name
        self.client_id = client_id
        self.mpc_paras = mpc_paras
        self.configs = configs

        self.task_start_time = time.time()


        # Make Log and Data directories under task directory
        try:
            os.mkdir(Config.TaskRootPath + task_name + "-%d" % client_id + "/log")
            self.task_log_path = Config.TaskRootPath + task_name + "-%d" % client_id + "/log/"
        except:
            self.logger.logE("Cannot create task log directory with root path {} and task_name {}".
                             format(Config.TaskRootPath, task_name))
        try:
            os.mkdir(Config.TaskRootPath + task_name + "-%d" % client_id + "/Data")
            self.task_data_path = Config.TaskRootPath + task_name + "-%d" % client_id + "/Data/"
        except:
            self.logger.logE("Cannot create task Data directory with root path {} and task_name {}".
                             format(Config.TaskRootPath, task_name))

        # Create Communication channel
        try:
            self.channel = Channel(client_id, "0.0.0.0:" + str(listen_port), len(ip_dict), ip_dict, time_out=30,
                                   logger=Logger(open(self.current_task_path + "log/channel_log.txt", "w+"), level=log_level))
        except:
            self.logger.logE("Cannot create channel with port {}, ip_dict {}".format(listen_port, ip_dict))

        if add_query_service_to_computation_grpc_server(self):
            self.logger.log("Attach query service to computation grpc server. Available to query stask status")
        else:
            self.logger.logW("Cannot attach query service since compuation server is not grpc server.")
        self.stage = TaskStage.Created


    def __load_data(self):
        try:
            self.train_data_loader = CSVDataLoader(self.preprocess_client.train_data_path, None, None)
            self.test_data_loader = CSVDataLoader(self.preprocess_client.test_data_path, None, None)
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
                                 self.current_task_path + "Data/",
                                 self.main_client, self.other_data_clients,
                                 Logger(open(self.current_task_path + "log/preprocess_log.txt", "w+"),
                                        level=self.log_config["log_level"]))

            def align_thread():
                if self.preprocess_client.start_align():
                    self.stage = TaskStage.LoadingData
                else:
                    self.stage = TaskStage.Error
        elif self.role is "main_client":
            self.preprocess_client = \
                MainPreprocessor(self.channel, self.data_clients,
                                 Logger(open(self.current_task_path + "log/preprocess_log.txt", "w+"),
                                        level=self.log_config["log_level"]))
            def align_thread():
                if self.preprocess_client.start_align():
                    self.stage = TaskStage.LoadingData
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
                self.train_client = FeatureClient(self.channel, self.train_data_loader, self.test_data_loader,
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
            from Client.SharedNN.ComputationProviders import MainClient
            self.train_client = MainClient(self.channel, self.data_clients, self.label_client, self.train_config,
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


def add_query_service_to_computation_grpc_server(task: MPCTask):
    from Communication.RPCComm import Peer
    peer = task.channel
    if not isinstance(peer, Peer):
        return False
    message_pb2_grpc.add_QueryMPCTaskServicer_to_server(QueryMPCTaskServicer(task), peer.server.server)
    return True
