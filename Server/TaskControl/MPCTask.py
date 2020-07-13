import time
import enum
import threading
import os
import pickle
import json
import re

from Client.Client import BaseClient
from Client.MPCClient import MPCClientParas
from Client.SharedNN.DataProviders import FeatureClient, LabelClient
from Server.TaskControl import TaskConfig
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


class MPCModels:
    models = {"shared_nn", "shared_lr"}
    SharedNN = "shared_nn"
    SharedLR = "shared_lr"


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


def str_dict_to_int(str_dict: dict):
    int_dict = dict()
    for key_str in str_dict:
        value = str_dict[key_str]
        int_dict[int(key_str)] = value
    return int_dict


class MPCTask:
    def __init__(self, role: str, task_name: str,
                 client_id: int, mpc_paras,
                 ip_dict: dict, listen_port: int,
                 configs: dict
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

        :param configs
        """
        if role not in MPCRoles.roles:
            self.logger.logE("Not a valid mpc role: {}".format(role))
            self.stage = TaskStage.Error
        self.role = role

        self.current_task_path = Config.TaskRootPath + task_name + "-%d" % client_id + "/"
        self.stage = TaskStage.Creating
        self.log_level = TaskConfig.TaskLogLevel
        self.logger = Logger(open(self.current_task_path + "task_log.txt", "w+"), level=self.log_level)

        # Since ip_dict maybe serialized from json. The key of ip_dict has to be int.
        self.ip_dict = str_dict_to_int(ip_dict)

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

        self.task_manage_client = BaseClient(self.channel, None)
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
                                   logger=Logger(open(self.current_task_path + "log/channel_log.txt", "w+"), level=self.log_level))
        except:
            self.logger.logE("Cannot create channel with port {}, ip_dict {}".format(listen_port, ip_dict))

        if add_query_service_to_computation_grpc_server(self):
            self.logger.log("Attach query service to computation grpc server. Available to query stask status")
        else:
            self.logger.logW("Cannot attach query service since compuation server is not grpc server.")
        self.stage = TaskStage.Created

    def start(self):
        raise NotImplementedError()


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
        }

    def _query_stage(self):
        return self.task.stage.name

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
