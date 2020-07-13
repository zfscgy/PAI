import numpy as np
import threading
import pathlib
import pickle
import time
from Client.Client import BaseClient, ClientException
import Client.MPCClient as MPCC
from Communication.Message import MessageType, ComputationMessage
from Communication.Channel import BaseChannel
from Client.Data.DataLoader import DataLoader
from Client.Learning.Losses import MSELoss
from Utils.Log import Logger


class FeatureClient(MPCC.DataClient):
    def __init__(self, channel: BaseChannel, logger: Logger, 
                 mpc_paras: MPCC.MPCClientParas, mpc_mode: MPCC.MPCClientMode,
                 data_loader: DataLoader, test_data_loader: DataLoader):
        # random generate some to data
        super(FeatureClient, self).__init__(channel, logger, mpc_paras, mpc_mode, data_loader, test_data_loader)
        self.other_feature_client_ids = self.feature_client_ids.copy()
        self.other_feature_client_ids.remove(self.client_id)


        self.batch_size = None
        self.n_rounds = 0

        self.batch_data = None
        self.para = None
        self.other_paras = [None for _ in range(channel.n_clients)]
        self.learning_rate = None

        self.error = False

        # 变量储存器，用于Secret Sharing矩阵乘法
        self.current_triplets = [None for _ in range(channel.n_clients)]
        self.shared_own_mat = [None for _ in range(channel.n_clients)]
        self.shared_other_mat = [None for _ in range(channel.n_clients)]
        self.recovered_own_value = [None for _ in range(channel.n_clients)]
        self.recovered_other_value = [None for _ in range(channel.n_clients)]
        self.shared_out_AB = [None for _ in range(channel.n_clients)]
        self.shared_out_BA = [None for _ in range(channel.n_clients)]
        self.own_out = None

        self.logger.log("Client initialized")

    def __calculate_first_hidden_layer(self, other_id):
        """
        This function is to interactively calculate matrix product with dataclient 'otherid'
        It is following the SMC protocol, the multiplication triplet is provided by a third service.
        :param other_id:
        :return:
        """

        # 提供数据作为矩阵乘法中的乘数
        def set_triplet_AB():
            self.send_check_msg(self.crypto_producer_id,
                          ComputationMessage(MessageType.SET_TRIPLET, (1, other_id, self.batch_data.shape,
                                                                       self.para.shape)))

        # 提供参数作为矩阵乘法中的的被乘数
        def set_triplet_BA():
            self.send_check_msg(self.crypto_producer_id,
                          ComputationMessage(MessageType.SET_TRIPLET, (2, other_id, self.other_paras[other_id].shape,
                                                                       (self.batch_data.shape[0], self.other_paras[other_id].shape[0]))))

        def get_triples():
            msg = self.receive_check_msg(self.crypto_producer_id, MessageType.TRIPLE_ARRAY, key=other_id)
            self.current_triplets[msg.data[0]] = msg.data[1:]

        def share_data():
            self.shared_own_mat[other_id] = self.batch_data * np.random.uniform(0, 1, self.batch_data.shape)
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_DATA_SHARE, self.batch_data - self.shared_own_mat[other_id]))

        def share_para():
            self.shared_own_mat[other_id] = self.other_paras[other_id] * \
                                            np.random.uniform(0, 1, self.other_paras[other_id].shape)
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_DATA_SHARE, self.other_paras[other_id] - self.shared_own_mat[other_id]))

        def get_other_share():
            other_share = self.receive_check_msg(other_id, MessageType.MUL_DATA_SHARE)
            self.shared_other_mat[other_id] = other_share.data

        def recover_own_value():
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_OwnVal_SHARE,
                                                       self.shared_own_mat[other_id] - self.current_triplets[other_id][0]))

        def get_other_value_share():
            msg = self.receive_check_msg(other_id, MessageType.MUL_OwnVal_SHARE)
            self.recovered_other_value[other_id] = self.shared_other_mat[other_id] - self.current_triplets[other_id][1] + msg.data

        def recover_other_value():
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_OtherVal_SHARE,
                                                       self.shared_other_mat[other_id] - self.current_triplets[other_id][1]))

        def get_own_value_share():
            msg = self.receive_check_msg(other_id, MessageType.MUL_OtherVal_SHARE)
            self.recovered_own_value[other_id] = self.shared_own_mat[other_id] - self.current_triplets[other_id][0] + msg.data

        def get_shared_out_AB():
            self.shared_out_AB[other_id] = - np.matmul(self.recovered_own_value[other_id],
                                                       self.recovered_other_value[other_id])
            self.shared_out_AB[other_id] += np.matmul(self.shared_own_mat[other_id], self.recovered_other_value[other_id]) + \
                                            np.matmul(self.recovered_own_value[other_id], self.shared_other_mat[other_id]) + self.current_triplets[other_id][2]

        def get_shared_out_BA():
            self.shared_out_BA[other_id] = np.matmul(self.recovered_other_value[other_id], self.shared_own_mat[other_id]) + \
                                           np.matmul(self.shared_other_mat[other_id], self.recovered_own_value[other_id]) + self.current_triplets[other_id][2]

        # Calculate X_own * Theta_other
        def calc_AB():

            set_triplet_AB()
            get_triples()

            share_data()
            get_other_share()
            recover_own_value()
            get_other_value_share()
            recover_other_value()
            get_own_value_share()
            get_shared_out_AB()

        # Calculate Theta_own * X_other
        def calc_BA():

            set_triplet_BA()
            get_triples()

            share_para()
            get_other_share()
            recover_own_value()
            get_other_value_share()
            recover_other_value()
            get_own_value_share()
            get_shared_out_BA()
        try:
            if other_id < self.client_id:
                calc_AB()
                calc_BA()
            else:
                calc_BA()
                calc_AB()
        except ClientException as e:
            self.logger.logE("Client Exception encountered, stop calculating.")
            self.error = True
        except Exception as e:
            self.logger.logE("Python Exception encountered , stop calculating.")
            self.error = True
        finally:
            return

    def __calc_out_share(self):
        calc_threads = []
        for client in self.other_feature_client_ids:
            calc_threads.append(threading.Thread(target=self.__calculate_first_hidden_layer, args=(client,)))
            calc_threads[-1].start()

        # While do secret-sharing matrix multiplication with other clients, do matrix multiplication on
        # local data and parameter.

        self.own_out = np.matmul(self.batch_data, self.para)

        for calc_thread in calc_threads:
            calc_thread.join()

    def __send_updates_to(self, client_id: int, update: np.ndarray):
        try:
            self.send_check_msg(client_id, ComputationMessage(MessageType.CLIENT_PARA_UPDATE, update))
        except:
            self.logger.logE("Error encountered while sending parameter updates to other data clients")
            self.error = True

    def __recv_updates_from(self, client_id: int):
        try:
            update_msg = self.receive_check_msg(client_id, MessageType.CLIENT_PARA_UPDATE)
            self.other_paras[client_id] -= self.learning_rate * update_msg.data
        except:
            self.logger.logE("Error encountered while receiving parameter updates from other data clients")
            self.error = True

    def __parameter_update(self):
        updates = self.receive_check_msg(self.main_client_id, MessageType.CLIENT_OUT_GRAD).data
        own_para_grad = self.batch_data.transpose() @ updates
        portion = np.random.uniform(0, 1, len(self.other_feature_client_ids) + 1)
        portion /= np.sum(portion)
        self.para -= self.learning_rate * own_para_grad * portion[0]
        send_update_threads = []
        for i, data_client in enumerate(self.other_feature_client_ids):
            send_update_threads.append(threading.Thread(
                target=self.__send_updates_to, args=(data_client, own_para_grad * portion[i + 1])
            ))
            send_update_threads[-1].start()

        recv_update_threads = []
        for data_client in self.other_feature_client_ids:
            recv_update_threads.append(threading.Thread(
                target=self.__recv_updates_from, args=(data_client,)
            ))
            recv_update_threads[-1].start()
        for thread in send_update_threads + recv_update_threads:
            thread.join()

    def __train_one_round(self):
        """

        :return: `True` if no error occurred during this training round. `False` otherwise
        """
        try:
            """
            Waiting for server's next-round message
            """
            start_msg = self.receive_check_msg(self.main_client_id,
                                               [MessageType.NEXT_TRAIN_ROUND, MessageType.TRAINING_STOP])
            """
            If server's message is stop message, stop
            otherwise, if server's next-round message's data is "Test", switch to test mode
            """
            if start_msg.header == MessageType.TRAINING_STOP:
                self.logger.log("Received server's training stop message, stop training")
                return False
            else:
                self.mpc_mode = MPCC.MPCClientMode.Train
                if start_msg.data == "Test":
                    self.logger.log("Received server test signal. Start test Round:")
                    self.mpc_mode = MPCC.MPCClientMode.Test

        except:
            self.logger.logE("Error encountered while receiving server's start message")
            self.error = True
            return False

        """
        Load batch data. If client is in test_mode, load from test data loader. 
        Otherwise, load from train data loader
        """
        if self.mpc_mode == MPCC.MPCClientMode.Train:
            self.batch_data = self.train_data_loader.get_batch(self.batch_size)
        else:
            self.batch_data = self.test_data_loader.get_batch(self.test_batch_size)
        """
        Interactively calculate first layer's output. Each data client gets a share of output o_i
        o_1 + o_2 + ... + o_n (elements-wise add) is the actual first layer's output
        """
        self.__calc_out_share()
        if self.error:
            self.logger.logE("Error encountered while calculating shared outputs")
            return False
        """
        Send the first layer's share to server
        """
        try:
            self.send_check_msg(self.main_client_id,
                                ComputationMessage(MessageType.MUL_OUT_SHARE,
                                                   (self.own_out, self.shared_out_AB, self.shared_out_BA)))
        except:
            self.logger.logE("Error encountered while sending output shares to server")
            self.error = True
            return False
        """
        If not in the test_mode, interactively calculate the gradients w.r.t. to data client's share of parameters
        """
        if self.mpc_mode != MPCC.MPCClientMode.Test:
            try:
                self.__parameter_update()
            except:
                self.logger.logE("Error encountered while updateing parameters")
                self.error = True
                return False

            if self.error:
                self.logger.logE("Error encountered while updateing parameters")
                return False

        return True

    def __set_config(self, config: dict):
        self.configs = config
        client_dims = config["client_dims"]
        out_dim = config["out_dim"]
        self.batch_size = config["batch_size"]
        self.test_batch_size = config.get("test_batch_size")
        self.learning_rate = config["learning_rate"]
        self.train_data_loader.set_random_seed(config["random_seed"])
        self.test_data_loader.set_random_seed(config["random_seed"])
        for other_id in client_dims:
            self.other_paras[other_id] = \
                np.random.normal(0,
                                 1 / (len(self.other_feature_client_ids) * client_dims[other_id]),
                                 [client_dims[other_id], out_dim])

    def load_parameters(self, directory):
        self.para = np.load(pathlib.Path(directory).joinpath("own_param.npy"))
        self.other_paras = pickle.load(pathlib.Path(directory).joinpath("other_paras.pkl"))

    def save(self, directory):
        np.save(pathlib.Path(directory).joinpath("own_param.npy"), self.para)
        pickle.dump(self.other_paras, pathlib.Path(directory).joinpath("other_paras.pkl"))

    def start_train(self, configs:dict=None):
        """
        :param wait_for_server:
        :return:
        """
        if configs is None:
            configs = dict()

        """
        Receive config message from server, then initialize some parameters
        After this, send CLIENT_READY message to server
        """
        self.logger.log("Client started, waiting for server config message with time out %.2f" % configs.get('wait_for_server'))
        try:
            msg = self.receive_check_msg(self.main_client_id, MessageType.TRAIN_CONFIG, time_out=configs.get('wait_for_server'))
            self.__set_config(msg.data)

            self.para = self.other_paras[self.client_id]
            self.send_check_msg(self.main_client_id, ComputationMessage(MessageType.CLIENT_READY, None))

        except Exception:
            self.logger.logE("Train not started due to error while receiving config message and sending client-ready")
            self.error = True
            return False

        self.logger.log("Received train conifg message: %s" % msg.data)

        self.n_rounds = 0
        while True:
            train_res = self.__train_one_round()
            self.n_rounds += 1
            self.logger.log("Train round %d finished" % self.n_rounds)
            """
            After one train round over, send CLIENT_ROUND_OVER message to server
            """
            try:
                self.send_check_msg(self.main_client_id, ComputationMessage(MessageType.CLIENT_ROUND_OVER, train_res))
            except:
                self.logger.logE("Error encountered while sending round over message to server")
                self.error = True
                return False
            if not train_res:
                if self.error:
                    self.logger.logE("Error encountered while training one round. Stop.")
                    return False
                else:
                    return True


class LabelClient(MPCC.DataClient):
    def __init__(self, channel: BaseChannel, logger:Logger,
                 mpc_paras: MPCC.MPCClientParas, mpc_mode: MPCC.MPCClientMode,
                 label_loader: DataLoader, test_label_loader: DataLoader,
                 loss_func=None, metric_func=None):
        super(LabelClient, self).__init__(channel, logger, mpc_paras, mpc_mode, label_loader, test_label_loader)
        self.batch_size = None
        self.test_batch_size = None
        if loss_func is None:
            self.loss_func = MSELoss()
        else:
            self.loss_func = loss_func
        if metric_func is None:
            self.metric_func = loss_func
        else:
            self.metric_func = metric_func

        self.n_rounds = 0
        self.start_time = 0
        self.metrics_record = []

        self.error = False

        # cached labels
        self.batch_labels = None

        self.logger.log("Client initialized")

    def __compute_pred_grad(self):
        preds = self.receive_check_msg(self.main_client_id, MessageType.PRED_LABEL).data
        loss = self.loss_func.forward(self.batch_labels, preds)
        metric = self.metric_func(self.batch_labels, preds)
        if self.mpc_mode == MPCC.MPCClientMode.Test:
            self.metrics_record.append([time.time() - self.start_time, self.n_rounds] + metric)
        self.logger.log("Current batch loss: {}, metric value: {}".format(loss, metric))
        grad = self.loss_func.backward()
        self.send_check_msg(self.main_client_id, ComputationMessage(MessageType.PRED_GRAD, (grad, loss)))

    def __train_one_round(self):
        try:
            start_msg = self.receive_check_msg(self.main_client_id,
                                               [MessageType.NEXT_TRAIN_ROUND, MessageType.TRAINING_STOP])
            if start_msg.header == MessageType.TRAINING_STOP:
                self.logger.log("Received server's training stop message, stop training")
                return False
            else:
                self.mpc_mode = MPCC.MPCClientMode.Train
                if start_msg.data in ["Test", "Predict"]:
                    self.logger.log("Test Round:")
                    self.mpc_mode = MPCC.MPCClientMode.Test
        except:
            self.logger.logE("Error encountered while receiving server's start message")
            self.error = True
            return False

        try:
            if self.mpc_mode != MPCC.MPCClientMode.Test:
                self.batch_labels = self.train_data_loader.get_batch(self.batch_size)
            else:
                self.batch_labels = self.test_data_loader.get_batch(self.test_batch_size)
        except:
            self.logger.logE("Error encountered while loading batch labels")
            self.error = True
            return False

        try:
            self.__compute_pred_grad()
        except:
            self.logger.logE("Error encountered while computing prediction gradients")
            return False
        return True

    def set_config(self, config: dict):
        self.batch_size = config["batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.train_data_loader.set_random_seed(config["random_seed"])
        self.test_data_loader.set_random_seed(config["random_seed"])

    def start_train(self, wait_for_server: float=100):
        """
        :param wait_for_server:
        :return:
        """
        self.logger.log("Client started, waiting for server config message with time out %.2f" % wait_for_server)
        """
        Receive config message from server, then initialize some parameters
        After this, send CLIENT_READY message to server
        """
        try:
            msg = self.receive_check_msg(self.main_client_id, MessageType.TRAIN_CONFIG, time_out=wait_for_server)
            self.set_config(msg.data)
            self.send_check_msg(self.main_client_id, ComputationMessage(MessageType.CLIENT_READY, None))
            self.start_time = time.time()
        except Exception as e:
            self.logger.logE("Python Exception encountered, stop\n")
            self.logger.logE("Train not started")
            return False

        self.logger.log("Received train config message: %s" % msg.data)

        self.n_rounds = 0
        while True:
            train_res = self.__train_one_round()
            self.n_rounds += 1
            self.logger.log("Train round %d finished" % self.n_rounds)
            try:
                self.send_check_msg(self.main_client_id, ComputationMessage(MessageType.CLIENT_ROUND_OVER, train_res))
            except:
                self.logger.logE("Error encountered while sending round over message to server")
                return False
            if not train_res:
                if self.error:
                    self.logger.logE("Error encountered while training one round. Stop.")
                    return False
                else:
                    return True
