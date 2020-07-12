import os

from Server.TaskControl.MPCTask import MPCTask, TaskStage, MPCClientParas, MPCRoles
from Client.Preprocess.PreprocessClient import PreprocessClient
from Client.SharedNN.DataProviders import FeatureClient, LabelClient
from Client.Data.DataLoader import CSVDataLoader
from Client.MPCClient import MPCClientMode
from Client.Learning.Losses import get_loss
from Communication.Message import ComputationMessage, MessageType
from Utils.Log import Logger


class DataProviderTask(MPCTask):
    def __init__(self, role: str, task_name: str,
                 client_id: int, mpc_paras: MPCClientParas,
                 ip_dict: dict, listen_port: int,
                 log_level: int, configs: dict):
        super(DataProviderTask, self).__init__(role, task_name, client_id, mpc_paras,
                                               ip_dict, listen_port, log_level, configs)
        if role not in [MPCRoles.FeatureProvider, MPCRoles.LabelProvider]:
            self.logger.logE("Role is not data_provider, task stop")
            self.stage = TaskStage.Error

        # Check configs
        if self.configs.get("file_path") is None:
            self.logger.logE("The configs must contain key file_path")
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
                self.logger.logE("The configs must contain key loss_func")
                self.stage = TaskStage.Error
                return
            self.loss_func = get_loss(self.configs["loss_func"])
            if self.loss_func is None:
                self.logger.logE("The loss_func is invalid")
                self.stage = TaskStage.Error
                return

        self.preprocess_client = PreprocessClient(
            self.channel, Logger(open(self.task_log_path + "channel_log.txt", "w+"), self.log_level),
            mpc_paras, self.configs["file_path"], self.task_data_path)
        self.train_client = None

    def start(self):
        self.stage = TaskStage.Proprecessing
        if self.preprocess_client.start_align():
            self.logger.log("Align finished. Start loading train and test data.")
        else:
            self.logger.logE("Align failed. Task stop.")
            return False
        self.stage = TaskStage.LoadingData

        try:
            train_data_loader = CSVDataLoader(self.preprocess_client.train_data_path)
            test_data_loader = CSVDataLoader(self.preprocess_client.test_data_path)
        except Exception as e:
            self.logger.logE("Loading data failed with error {}. Task stop.".format(e))
            return False

        self.logger.log("Loading data finished. Start training")
        if self.role == MPCRoles.FeatureProvider:
            self.train_client = FeatureClient(
                self.channel, Logger(open(self.task_log_path + 'train_log.txt', "w+"), level=self.log_level),
                self.mpc_paras, MPCClientMode.Train, train_data_loader, test_data_loader)
        else:
            self.train_client = FeatureClient(
                self.channel, Logger(open(self.task_log_path + 'train_log.txt', "w+"), level=self.log_level),
                self.mpc_paras, MPCClientMode.Train, train_data_loader, test_data_loader)
        try:
            self.train_client.send_check_msg(self.mpc_paras.main_client_id,
                                             ComputationMessage(MessageType.NULL, "load-data-done"))
        except:
            self.logger.logE("Sending load-data-done message to main client failed")
            self.stage = TaskStage.Error
            return False
