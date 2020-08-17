import enum
from Client.MPCClient import MPCClientParas
from Utils.Log import Logger
import Task.TaskConfig as Config
from Task.ClientBuilder import build_client
from Task.TaskQuery import add_query_service_to_computation_grpc_server
from Task.Utils import str_dict_to_int
from Communication.RPCComm import Peer as Channel


class TaskStatus(enum.Enum):
    Error = 0
    Created = 1
    Running = 2
    Finished = 3


class Task:
    def __init__(self, task_name, client_id: int, client_port: int, ip_dict: dict, client_config: dict):
        self.status = None
        self.task_path = Config.TaskRootPath + task_name + "-%d" % client_id + "/"

        self.arg_dict = dict()

        self.arg_dict["logger"] = self.logger = Logger(open(self.task_path + "log.txt", "w"))
        self.arg_dict["task_path"] = self.task_path
        try:
            ip_dict = str_dict_to_int(ip_dict)
            self.arg_dict["channel"] = self.channel = \
                Channel(client_id, '127.0.0.1:%d' % client_port, 3, ip_dict, 120, self.logger)
        except:
            self.logger.logE("Failed to create channel. Abort.")
            self.status = TaskStatus.Error
            return

        try:
            self.arg_dict["mpc_paras"] = MPCClientParas(**client_config["mpc_paras"])
            del client_config["mpc_paras"]
            self.arg_dict.update(client_config)
        except:
            self.logger.logE("Set mpc_paras parameter failed. Abort.")
            self.status = TaskStatus.Error
            return

        try:
            self.client_handle = build_client(self.arg_dict)
        except:
            self.logger.logE("Failed to build client. Abort.")
            self.status = TaskStatus.Error
            return

        grpc_servicer = add_query_service_to_computation_grpc_server(self)
        if grpc_servicer is not None:
            self.logger.log("Attach query service to computation grpc server. Available to query stask status")
        else:
            self.logger.logW("Cannot attach query service since compuation server is not grpc server.")
        self.grpc_servicer = grpc_servicer
        if self.grpc_servicer is not None:
            self.grpc_servicer.add_query_dict(self.client_handle.calls)
            self.grpc_servicer.add_query("status", lambda: self.status.name)

        self.status = TaskStatus.Created

    def start(self):
        if self.status == TaskStatus.Error:
            self.logger.logE("Cannot start a error task.")
            open(self.task_path + "failed", "w").close()
            return False
        if self.client_handle.start():
            open(self.task_path + "finished", "w").close()
            return True
        else:
            open(self.task_path + "failed", "w").close()
            return False
