import threading
from Client.MPCClient import MPCClientParas, MPCClientMode
from Client.Preprocess.PreprocessClient import MainPreprocessor
from Client.SharedNN.ComputationProviders import MainClient
from Server.TaskControl.MPCTask import MPCTask, str_dict_to_int, TaskStage, TaskQueryStatus
from Utils.Log import Logger


class MainProviderTask(MPCTask):
    def __init__(self, role: str, task_name: str, model_name: str,
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
        super(MainProviderTask, self).__init__(role, task_name, model_name, client_id, mpc_paras,
                                               ip_dict, listen_port, configs)

        if configs.get("train_config") is None:
            self.logger.logE("Configs must include train_config")
            self.stage = TaskStage.Error
            return
        else:
            self.train_config = configs.get("train_config")

        self.configs["train_config"]["client_dims"] = str_dict_to_int(self.configs["train_config"]["client_dims"])

        self.preprocess_client = MainPreprocessor(
            self.channel, Logger(open(self.task_log_path + "preprocess_log.txt", "w+"), level=self.log_level), self.mpc_paras)
        self.train_client = None

        if self.grpc_servicer is not None:
            self.grpc_servicer.add_query("n_batches", lambda: (TaskQueryStatus.err, "Train not started"))

    def start(self):
        super(MainProviderTask, self).start()
        if self.preprocess_client.start_align():
            self.logger.log("Aligned finished. Wait for data clients load data.")
        else:
            self.logger.logE("Align failed. Train stop.")
            return False

        self.train_client = MainClient(
            self.channel, Logger(open(self.task_log_path + "main_client_log.txt", "w+"), level=self.log_level),
            self.mpc_paras, MPCClientMode.Train, self.train_config)
        if self.grpc_servicer is not None:
            self.grpc_servicer.add_query("n_batches", lambda: (TaskQueryStatus.ok, self.train_client.n_rounds))

        self.stage = TaskStage.Training
        if self.train_client.start_train():
            self.logger.logE("Train failed. Train stop.")
            self.stage = TaskStage.Error
            return False
        else:
            self.logger.log("Train finished")
            self.stage = TaskStage.Finished
