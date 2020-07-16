import time
import enum
import os
import pickle

from Client.Client import BaseClient
from Client.MPCClient import MPCClientParas
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
    Training = 3
    Finished = 4
    Error = 5

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
    def __init__(self, role: str, task_name: str, model_name: str,
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
        self.current_task_path = Config.TaskRootPath + task_name + "-%d" % client_id + "/"
        self.log_level = Config.TaskLogLevel
        self.logger = Logger(open(self.current_task_path + "task_log.txt", "w+"), level=self.log_level)
        self.stage = TaskStage.Creating


        if role not in MPCRoles.roles:
            self.logger.logE("Not a valid mpc role: {}".format(role))
            self.stage = TaskStage.Error
            return
        self.role = role

        if model_name not in MPCModels.models:
            self.logger.logE("Not a valid mpc model: {}".format(model_name))
            self.stage = TaskStage.Error
            return
        self.model_name = model_name

        # Make Log and Data directories under task directory
        try:
            os.mkdir(self.current_task_path + "log")
            self.task_log_path = self.current_task_path + "log/"
        except:
            self.logger.logE("Cannot create task log directory with root path {} and task_name {}".
                             format(Config.TaskRootPath, task_name))
            self.stage = TaskStage.Error
            return

        try:
            os.mkdir(self.current_task_path + "Data")
            self.task_data_path = self.current_task_path + "Data/"
        except:
            self.logger.logE("Cannot create task Data directory with root path {} and task_name {}".
                             format(Config.TaskRootPath, task_name))
            self.stage = TaskStage.Error
            return

        # Since ip_dict maybe serialized from json. The key of ip_dict has to be int.
        self.ip_dict = str_dict_to_int(ip_dict)

        if isinstance(mpc_paras, dict):
            try:
                mpc_paras = MPCClientParas(**mpc_paras)
            except:
                self.logger.logE("mpc_paras not correct")
                self.stage = TaskStage.Error
                return
        elif not isinstance(mpc_paras, MPCClientParas):
            self.logger.logE("mpc_paras must be a dict or a MPCClientParas instance")
            self.stage = TaskStage.Error

        self.name = task_name
        self.client_id = client_id
        self.mpc_paras = mpc_paras

        self.configs = configs

        # Create Communication channel
        try:
            self.channel = Channel(client_id, "0.0.0.0:" + str(listen_port), len(ip_dict), self.ip_dict, time_out=60,
                                   logger=Logger(open(self.current_task_path + "log/channel_log.txt", "w+"), level=self.log_level))
        except:
            self.logger.logE("Cannot create channel with port {}, ip_dict {}".format(listen_port, ip_dict))
            self.stage = TaskStage.Error

        self.task_start_time = time.time()

        self.task_manage_client = BaseClient(self.channel, self.logger)


        grpc_servicer = add_query_service_to_computation_grpc_server(self)
        if grpc_servicer is not None:
            self.logger.log("Attach query service to computation grpc server. Available to query stask status")
        else:
            self.logger.logW("Cannot attach query service since compuation server is not grpc server.")
        self.grpc_servicer = grpc_servicer
        if self.grpc_servicer is not None:
            self.grpc_servicer.add_query("stage", lambda: (TaskQueryStatus.ok, self.stage.name))

        self.stage = TaskStage.Created

    def start(self):
        if self.stage is TaskStage.Error:
            self.logger.logE("Cannot start a error task.")
            return False

class TaskQueryStatus(enum.Enum):
    ok = 0
    err = 1


class QueryMPCTaskServicer(message_pb2_grpc.QueryMPCTaskServicer):
    def __init__(self, task: MPCTask):
        self.task = task
        self.query_dict = {
            "QueryStage": self._query_stage,
        }

    def _query_stage(self):
        return self.task.stage.name

    def QueryTask(self, request: message_pb2.TaskQuery, context):
        def encode(status: TaskQueryStatus, obj=None):
            return message_pb2.TaskResponse(status=status.value, python_bytes=pickle.dumps(obj))
        if request.query_string not in self.query_dict:
            return encode(TaskQueryStatus.err, "Query url not exist")
        status, resp = self.query_dict[request.query_string]()
        return encode(status, resp)

    def add_query(self, url, func):
        if callable(func):
            self.query_dict[url] = func
            return True
        else:
            return False

def add_query_service_to_computation_grpc_server(task: MPCTask):
    from Communication.RPCComm import Peer
    peer = task.channel
    if not isinstance(peer, Peer):
        return None
    task_servicer = QueryMPCTaskServicer(task)
    message_pb2_grpc.add_QueryMPCTaskServicer_to_server(task_servicer, peer.server.server)
    return task_servicer
