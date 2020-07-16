import os

from Server.TaskControl.MPCTask import MPCTask, TaskStage, MPCClientParas, MPCRoles
from Server.TaskControl.TaskConfig import DataRootPath
from Client.Preprocess.PreprocessClient import PreprocessClient
from Client.SharedNN.DataProviders import FeatureClient, LabelClient
from Client.Data.DataLoader import CSVDataLoader
from Client.MPCClient import MPCClientMode
from Client.Learning.Losses import get_loss
from Client.Learning.Metrics import get_metric
from Utils.Log import Logger


class DataProviderTask(MPCTask):
    def __init__(self, role: str, task_name: str, model_name: str,
                 client_id: int, mpc_paras: MPCClientParas,
                 ip_dict: dict, listen_port: int, configs: dict):
        """
        :param role:
        :param task_name:
        :param client_id:
        :param mpc_paras:
        :param ip_dict:
        :param listen_port:
        :param log_level:
        :param configs:
                Should be like
                {
                    "data_file": xxx,
                    // For both
                    "train_configs": {
                        "wait_for_server": 100,
                        // Only for label client:
                        "loss_func": xxx,
                        "metrics": xxx,
                    }
                {
        """
        super(DataProviderTask, self).__init__(role, task_name, model_name, client_id, mpc_paras,
                                               ip_dict, listen_port, configs)
        if role not in [MPCRoles.FeatureProvider, MPCRoles.LabelProvider]:
            self.logger.logE("Role is not data_provider, task stop")
            self.stage = TaskStage.Error
            return
        # Check configs
        if self.configs.get("data_file") is None:
            self.logger.logE("The configs must contain key data_file")
            self.stage = TaskStage.Error
            return

        if self.role is MPCRoles.LabelProvider:
            if self.configs.get("loss_func") is None:
                self.logger.logE("The configs must contain key loss_func")
                self.stage = TaskStage.Error
                return
            self.loss_func = get_loss(self.configs["loss_func"])
            if self.loss_func is None:
                self.logger.logE("The loss_func is invalid")
                self.stage = TaskStage.Error
                return

            if self.configs.get("metrics") is None:
                self.logger.logE("The configs must contain key metrics")
                self.stage = TaskStage.Error
                return
            self.metrics = get_metric(self.configs["metrics"])
            if self.loss_func is None:
                self.logger.logE("The metrics is invalid")
                self.stage = TaskStage.Error
                return

        self.preprocess_client = PreprocessClient(
            self.channel, Logger(open(self.task_log_path + "preprocess_log.txt", "w+"), level=self.log_level),
            self.mpc_paras, DataRootPath + self.configs["data_file"], self.task_data_path)
        self.train_client = None


    def start(self):
        super(DataProviderTask, self).start()
        self.stage = TaskStage.Proprecessing
        if self.preprocess_client.start_align():
            self.logger.log("Align finished. Start loading train and test data.")
        else:
            self.logger.logE("Align failed. Task stop.")
            return False

        try:
            train_data_loader = CSVDataLoader(self.preprocess_client.train_data_path)
            test_data_loader = CSVDataLoader(self.preprocess_client.test_data_path)
        except Exception as e:
            self.logger.logE("Loading data failed with error {}. Task stop.".format(e))
            return False

        self.logger.log("Loading data finished. Start training")
        self.stage = TaskStage.Training
        if self.role == MPCRoles.FeatureProvider:
            self.train_client = FeatureClient(
                self.channel, Logger(open(self.task_log_path + 'train_log.txt', "w+"), level=self.log_level),
                self.mpc_paras, MPCClientMode.Train, train_data_loader, test_data_loader)
        else:
            self.train_client = LabelClient(
                self.channel, Logger(open(self.task_log_path + 'train_log.txt', "w+"), level=self.log_level),
                self.mpc_paras, MPCClientMode.Train, train_data_loader, test_data_loader,
                self.loss_func, self.metrics)

        if self.train_client.start_train():
            self.logger.log("Train finished.")
            self.stage = TaskStage.Finished
            # Should add some methods to store data
            return True
        else:
            self.logger.logE("Train failed.")
            self.stage = TaskStage.Error
        return False