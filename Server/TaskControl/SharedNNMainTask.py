import threading
from Client.MPCClient import MPCClientParas, MPCClientMode
from Client.Preprocess.PreprocessClient import MainPreprocessor
from Client.SharedNN.ComputationProviders import MainClient
from Communication.Message import MessageType
from Server.TaskControl.MPCTask import MPCTask, str_dict_to_int, TaskStage
from Utils.Log import Logger


class MainMainTask(MPCTask):
    def __init__(self, role: str, task_name: str,
                 client_id: int, mpc_paras: MPCClientParas,
                 ip_dict: dict, listen_port: int,
                 configs: dict):
        """
        :param role:
        :param task_name:
        :param client_id:
        :param mpc_paras:
        :param ip_dict:
        :param listen_port:
        :param configs:
            {
                "data_load_time": 100
                train_config:
                {
                    "client_dims": {2: 20, 4: 30} or {"2": 20, "4": 30},
                    "out_dim: 1,
                    "layers": [],
                    "test_per_batch": 101,
                    "test_batch_size": None,
                    "learning_rate": 0.1,
                    "max_iter": 1002,
                }
            }
        """
        super(MainMainTask, self).__init__(role, task_name, client_id, mpc_paras,
                                           ip_dict, listen_port, configs)
        if configs.get("data_load_time") is None:
            self.logger.logE("Configs must include data_load_time")
            self.stage = TaskStage.Error
            return
        else:
            self.data_load_time = configs.get("data_load_time")

        if configs.get("train_config") is None:
            self.logger.logE("Configs must include train_config")
            self.stage = TaskStage.Error
            return
        else:
            self.train_config = configs.get("train_config")

        self.configs["train_config"]["client_dims"] = str_dict_to_int(self.configs["train_config"]["client_dims"])

        self.preprocess_client = MainPreprocessor(
            self.channel, Logger(open(self.task_log_path + "preprocess_log.txt", "w+"), level=log_level), mpc_paras)
        self.train_client = None

    def start(self):
        if self.preprocess_client.start_align():
            self.logger.log("Aligned finished. Wait for data clients load data.")
        else:
            self.logger.logE("Align failed. Train stop.")

        self.train_client = MainClient(
            self.channel, Logger(open(self.task_log_path + "main_client_log.txt", "w+"), level=self.log_level),
            self.mpc_paras, MPCClientMode.Train, self.train_config)
        self.stage = TaskStage.LoadingData
        data_client_ids = self.mpc_paras.feature_client_ids + [self.mpc_paras.label_client_id]
        not_ok_clients = []
        receiving_threads = []
        def receive_dataload_ok_from(client_id: int, not_ok_clients: list):
            try:
                self.task_manage_client.receive_check_msg(client_id, MessageType.NULL, self.data_load_time)
            except:
                self.logger.logE("Receive data load ok message from client %d failed" % client_id)
                not_ok_clients.append(client_id)

        for data_client_id in data_client_ids:
            receiving_threads.append(threading.Thread(
                target=receive_dataload_ok_from, args=(data_client_id, not_ok_clients), name="Receive-Load-Data-From-%d" % data_client_id))
            receiving_threads[-1].start()
        for receiving_thread in receiving_threads:
            receiving_thread.join()

        if len(not_ok_clients) > 0:
            self.logger.logE("Receive data-load-done message from data clients failed. Train stop.")
            self.stage = TaskStage.Error
            return False
        self.stage = TaskStage.Training
        if self.train_client.start_train():
            self.logger.logE("Train failed. Train stop.")
            self.stage = TaskStage.Error
            return False
        else:
            self.logger.log("Train finished")
            self.stage = TaskStage.Finished
