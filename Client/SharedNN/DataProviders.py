import numpy as np
import threading
import pathlib
import pickle
import time
import pandas as pd
import Client.MPCClient as MPCC
from Communication.Message import MessageType, PackedMessage
from Communication.Channel import BaseChannel
from Client.Data.DataLoader import DataLoader
from Client.Learning.Losses import MSELoss
from Client.Common.BroadcastClient import BroadcastClient
from Client.Common.SecureMultiplicationClient import SecureMultiplicationClient
from Utils.Log import Logger


class DataClient(MPCC.DataClient):
    def __init__(self, channel: BaseChannel, logger: Logger,
                 mpc_paras: MPCC.MPCClientParas,
                 data_loader: DataLoader, test_data_loader: DataLoader):
        super(DataClient, self).__init__(channel, logger, mpc_paras, data_loader, test_data_loader)

        self.other_feature_client_ids = self.feature_client_ids.copy()
        if self.client_id in self.other_feature_client_ids:
            self.other_feature_client_ids.remove(self.client_id)

        self.data_dim = self.data_shape[-1]
        self.batch_size = None
        self.n_rounds = 0
        self.batch_data = None
        self.broadcaster = BroadcastClient(self.channel, self.logger)

    def _before_training(self):
        self.logger.log("Start sync random seed with other data-providers.")
        random_seed = np.random.randint(0, 999999999)
        other_data_client_ids = self.feature_client_ids + [self.label_client_id]
        other_data_client_ids.remove(self.client_id)
        self.broadcaster.broadcast(other_data_client_ids,
                                   PackedMessage(MessageType.SharedNN_RandomSeed, random_seed))
        all_seeds = self.broadcaster.receive_all(other_data_client_ids, MessageType.SharedNN_RandomSeed)
        if not self.broadcaster.error:
            for c in all_seeds:
                random_seed ^= all_seeds[c]
            self.train_data_loader.set_random_seed(random_seed)
            self.test_data_loader.set_random_seed(random_seed)
            self.logger.log("Random seed swapped and set to data loaders.")
            return True
        else:
            self.logger.log("Swapping random seed with other data-providers failed. Stop training.")
            return False


class FeatureClient(DataClient):
    def __init__(self, channel: BaseChannel, logger: Logger,
                 mpc_paras: MPCC.MPCClientParas,
                 data_loader: DataLoader, test_data_loader: DataLoader):
        # random generate some to data
        super(FeatureClient, self).__init__(channel, logger, mpc_paras, data_loader, test_data_loader)

        self.batch_data = None
        self.para = None
        self.other_paras = dict()
        self.learning_rate = None

        self.error = False
        self.finished = False

        self.shared_out_AB = dict()
        self.shared_out_BA = dict()
        self.own_out = None

        self.multipliers = dict()
        for other_id in self.other_feature_client_ids:
            self.multipliers[other_id] = SecureMultiplicationClient(self.channel, self.logger)

        self.logger.log("Client initialized")

    def _before_training(self):
        if not super(FeatureClient, self)._before_training():
            return False
        try:
            self.send_check_msg(self.main_client_id, PackedMessage(MessageType.SharedNN_ClientDim,
                                                                   self.data_dim))
            config = self.receive_check_msg(self.main_client_id, MessageType.SharedNN_TrainConfig).data
            self.logger.log("Received main client's config message: {}".format(config))
            self.other_feature_dims = config["client_dims"]
            self.output_dim = config["out_dim"]
            self.batch_size = config["batch_size"]
            self.test_batch_size = config["test_batch_size"]
            self.learning_rate = config["learning_rate"]
        except:
            self.logger.logE("Get training config from server failed. Stop training.")
            return False

        try:
            for client_id in self.other_feature_dims:
                other_dim = self.other_feature_dims[client_id]
                self.other_paras[client_id] = np.random.normal(0, 1 / (len(self.feature_client_ids) * other_dim),
                                                        [other_dim, self.output_dim])
        except:
            self.logger.logE("Initialize weights failed. Stop training.")
            return False
        self.para = np.random.normal(0, 1 / (len(self.feature_client_ids) * self.data_dim), [self.data_dim, self.output_dim])

        return True

    def _forward(self):
        if self.mpc_mode is MPCC.ClientMode.Train:
            self.batch_data = self.train_data_loader.get_batch(self.batch_size)
        else:
            self.logger.log("Test round: load data from test dataset")
            self.batch_data = self.test_data_loader.get_batch(self.test_batch_size)

        def thread_mul_with_client(client_id: int):
            multiplier = self.multipliers[client_id]
            if self.client_id < client_id:
                if not multiplier.multiply_AB_with(client_id, self.crypto_producer_id,
                                               (self.data_dim, self.output_dim), self.batch_data):
                    self.error = True
                    return
                self.shared_out_AB[client_id] = multiplier.product
                if not multiplier.multiply_BA_with(client_id, self.crypto_producer_id,
                                                   (self.batch_data.shape[0], self.other_feature_dims[client_id]),
                                                   self.other_paras[client_id]):
                    self.error = True
                    return
                self.shared_out_BA[client_id] = multiplier.product
            else:
                if not multiplier.multiply_BA_with(client_id, self.crypto_producer_id,
                                                   (self.batch_data.shape[0], self.other_feature_dims[client_id]),
                                                   self.other_paras[client_id]):
                    self.error = True
                    return
                self.shared_out_BA[client_id] = multiplier.product
                if not multiplier.multiply_AB_with(client_id, self.crypto_producer_id,
                                               (self.data_dim, self.output_dim), self.batch_data):
                    self.error = True
                    return
                self.shared_out_AB[client_id] = multiplier.product

        mul_threads = []
        for client_id in self.other_feature_client_ids:
            mul_threads.append(threading.Thread(target=thread_mul_with_client, args=(client_id,)))
            mul_threads[-1].start()
        for mul_thread in mul_threads:
            mul_thread.join()
        if self.error:
            self.logger.logE("Multiplication on shared parameters with other clients failed. Stop training.")
            return False

        self.own_out = self.batch_data @ self.para

        try:
            self.send_check_msg(self.main_client_id,
                                PackedMessage(MessageType.SharedNN_FeatureClientOut,
                                              (self.own_out, self.shared_out_AB, self.shared_out_BA)))
        except:
            self.logger.logE("Send outputs to main client failed. Stop training.")
            return False

        return True

    def _backward(self):
        try:
            grad_on_output, status = self.receive_check_msg(self.main_client_id, MessageType.SharedNN_FeatureClientGrad).data
        except:
            self.logger.logE("Receive grads message from main client failed. Stop training.")
            return False
        if grad_on_output is None:
            pass
        else:
            own_para_grad = self.batch_data.transpose() @ grad_on_output
            portion = np.random.uniform(0, 1, len(self.feature_client_ids))
            portion /= np.sum(portion)
            self.para -= self.learning_rate * own_para_grad * portion[0]
            other_grad_msgs = dict()
            current_portion = 1
            for other_id in self.other_feature_client_ids:
                other_grad_msgs[other_id] = PackedMessage(
                    MessageType.SharedNN_FeatureClientParaGrad, own_para_grad * portion[current_portion])
                current_portion += 1

            try:
                self.broadcaster.broadcast(self.other_feature_client_ids, other_grad_msgs)
                self_para_grads = self.broadcaster.receive_all(self.other_feature_client_ids, MessageType.SharedNN_FeatureClientParaGrad)
                for other_id in self_para_grads:
                    self.other_paras[other_id] -= self.learning_rate * self_para_grads[other_id]
            except:
                self.logger.logE("Swap parameter gradients with other clients failed. Stop training.")
                return False

        if status == "Stop":
            self.logger.log("Received Stop signal from main-client, stop training.")
            self.finished = True
        elif status == "Continue-Test":
            self.mpc_mode = MPCC.ClientMode.Test
        else:
            self.mpc_mode = MPCC.ClientMode.Train

        return True

    def start_train(self):
        if not self._before_training():
            return False
        while True:
            if not self._forward():
                return False
            if not self._backward():
                return False
            if self.finished:
                return True
            self.n_rounds += 1

    def load_parameters(self, directory):
        self.para = np.load(pathlib.Path(directory).joinpath("own_param.npy"))
        self.other_paras = pickle.load(pathlib.Path(directory).joinpath("other_paras.pkl"))

    def save_parameters(self, directory):
        np.save(pathlib.Path(directory).joinpath("own_param.npy"), self.para)
        pickle.dump(self.other_paras, pathlib.Path(directory).joinpath("other_paras.pkl"))


class LabelClient(DataClient):
    def __init__(self, channel: BaseChannel, logger: Logger,
                 mpc_paras: MPCC.MPCClientParas,
                 label_loader: DataLoader, test_label_loader: DataLoader,
                 loss_func, metric_func, task_dir: str):
        super(LabelClient, self).__init__(channel, logger, mpc_paras, label_loader, test_label_loader)


        self.loss_func = loss_func
        self.metric_func = metric_func

        self.n_rounds = 0
        self.start_time = 0
        self.test_record = []
        self.task_dir = task_dir

        self.error = False
        self.finished = False

    def _before_training(self):
        if not super(LabelClient, self)._before_training():
            return False
        try:
            config = self.receive_check_msg(self.main_client_id, MessageType.SharedNN_TrainConfig).data
            self.logger.log("Received main client's config message: {}".format(config))
            self.batch_size = config["batch_size"]
            self.test_batch_size = config["test_batch_size"]
        except:
            self.logger.logE("Get training config from server failed. Stop training.")
            return False
        self.start_time = time.time()
        return True

    def _compute_loss(self):
        try:
            preds, mode = self.receive_check_msg(self.main_client_id, MessageType.SharedNN_MainClientOut).data
            if mode == "Train" or mode == "Train-Stop":
                self.mpc_mode = MPCC.ClientMode.Train
                self.batch_data = self.train_data_loader.get_batch(self.batch_size)
            elif mode == "Test" or mode == "Test-Stop":
                self.mpc_mode = MPCC.ClientMode.Test
                self.logger.log("Received Test signal. Load test set data to batch.")
                self.batch_data = self.test_data_loader.get_batch(self.test_batch_size)
            if mode[-4:] == "Stop":
                self.logger.log("Received Stop signal. Stop training after finish this round.")
                self.finished = True

            loss = self.loss_func.forward(self.batch_data, preds)
            metric = self.metric_func(self.batch_data, preds)
            if self.mpc_mode is MPCC.ClientMode.Test:
                self.test_record.append([time.time() - self.start_time, self.n_rounds, loss] + metric)
            self.logger.log("Current batch: {} loss: {}, metric value: {}".format(self.n_rounds, loss, metric))
            grad = self.loss_func.backward()
            self.send_check_msg(self.main_client_id, PackedMessage(MessageType.SharedNN_MainClientGradLoss, (grad, loss)))

        except:
            self.logger.logE("Compute gradient for main client predictions failed. Stop training.")
            return False

        return True

    def start_train(self):
        if not self._before_training():
            return False
        while True:
            if not self._compute_loss():
                return False
            if self.finished:
                pd.DataFrame(self.test_record,
                             columns=["time", "n_batches", "loss"] + ["metrics"] * (len(self.test_record[0]) - 3)).\
                    to_csv(self.task_dir + "record.csv", index=False)
                return True
            self.n_rounds += 1
