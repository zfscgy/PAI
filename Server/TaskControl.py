import time
from Client.Data import CSVDataLoader
from Utils.Log import Logger

class TaskException(Exception):
    def __init__(self, msg):
        super(TaskException, self).__init__()
        self.msg = msg

    def __str__(self):
        return "ClientException:" + self.msg


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
                 data_clients: list, label_client: int, main_client: int, ip_dict: dict, listen_port: int,
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
        :param data_config: Example: { "train_data_path": filepath, "test_data_path": filepath }
        :param log_config: Example: { "channel_log": filepath, "client_log": filepath, "log_level": 0 }
        """
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
        self.label_client = label_client
        self.main_client = main_client
        self.ip_dict = ip_dict
        self.task_start_time = time.time()
        self.task_client = None
        self.data_config = data_config
        self.log_config = log_config

        try:
            self.channel = Channel(client_id, "0.0.0.0:" + str(listen_port), len(ip_dict),
                                   logger=Logger(open(log_config["channel_log"])))
        except:
            raise TaskException("Cannot create channel, maybe the logger config is wrong.")

        try:
            self.train_data_loader = CSVDataLoader(data_config["train_data_path"], None, None)
            self.test_data_loader = CSVDataLoader(data_config["test_data_path"], None, None)
            if self.test_data_loader.data.shape[1] != self.train_data_loader:
                raise TaskException("Train data shape and test data shape must match.")
        except:
            raise TaskException("Cannot create data loader, check data_config.")



def start_task(task: MPCTask):
    if task.role is "client":
        from Client.DataProviders import DataClient