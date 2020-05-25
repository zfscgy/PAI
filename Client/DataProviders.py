import numpy as np
import threading
import traceback
from Client.Client import BaseClient, ClientException
from Communication.Message import MessageType, ComputationMessage
from Communication.Channel import BaseChannel
from Client.Data import DataLoader
from Client.Learning.Losses import LossFunc, MSELoss
from Client.Learning.Metrics import onehot_accuracy
from Utils.Log import Logger


class TripletsProvider(BaseClient):
    """
    A client for generate beaver triples. It can listen to other clients
    """
    def __init__(self, channel: BaseChannel, logger:Logger=None):
        super(TripletsProvider, self).__init__(channel, logger)
        self.triplets_id = self.client_id
        self.triplet_proposals = dict()
        self.listening_thread = [None for _ in range(channel.n_clients)]
        self.listening = False

    def listen_to(self, sender: int):
        msg = self.receive_msg(sender)
        if msg is not None:
            if msg.header == MessageType.SET_TRIPLET:
                operand_sender, target, shape_sender, shape_target = msg.data
                existing_shapes = self.triplet_proposals.get((target, sender))
                if existing_shapes is not None:
                    if existing_shapes[1] == shape_target and existing_shapes[2] == shape_sender:
                        self.generate_and_send_triplets(existing_shapes[0], (target, sender), (shape_target, shape_sender))
                        del self.triplet_proposals[(target, sender)]
                    else:
                        self.logger.logW("Triplet shapes %s %s not match with clients %d and %d" % (shape_target, shape_sender, target, sender))
                else:
                    self.triplet_proposals[(sender, target)] = (operand_sender, shape_sender, shape_target)
            else:
                self.logger.logW("Expect SET_TRIPLET message, but received %s from %d" % (msg.header, sender))

    def generate_and_send_triplets(self, first_operand: int, clients, shapes):
        # 判断哪一个是乘数，哪一个是被乘数
        if first_operand == 2:
            shapes = [shapes[1], shapes[0]]
            clients = [clients[1], clients[0]]
        u0 = np.random.uniform(-1, 1, shapes[0])
        u1 = np.random.uniform(-1, 1, shapes[0])
        v0 = np.random.uniform(-1, 1, shapes[1])
        v1 = np.random.uniform(-1, 1, shapes[1])
        z = np.matmul(u0 + u1, v0 + v1)
        z0 = z * np.random.uniform(0, 1, z.shape)
        z1 = z - z0
        self.send_msg(clients[0], ComputationMessage(MessageType.TRIPLE_ARRAY, (clients[1], u0, v0, z0)))
        self.send_msg(clients[1], ComputationMessage(MessageType.TRIPLE_ARRAY, (clients[0], v1, u1, z1)))

    def listen_to_client(self, sender_id):
        while self.listening:
            self.listen_to(sender_id)

    def start_listening(self):
        """
        Start the listening thread
        """
        self.listening = True
        for i in range(self.channel.n_clients):
            self.listening_thread[i] = threading.Thread(target=self.listen_to_client, args=(i,))
            self.listening_thread[i].start()

    def stop_listening(self):
        """
        Stop the listening thread
        """
        self.listening = False
        for i in range(self.channel.n_clients):
            self.listening_thread[i].join()


class DataClient(BaseClient):
    def __init__(self, channel: BaseChannel, data_loader: DataLoader, test_data_loader: DataLoader,
                 server_id: int, triplets_id: int, other_data_clients: list, logger: Logger=None):
        # random generate some to data
        super(DataClient, self).__init__(channel, logger)
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.batch_data = None
        self.server_id = server_id
        self.triplets_id = triplets_id
        self.other_data_clients = other_data_clients

        # Configs
        self.batch_size = None
        self.para = None
        self.other_paras = [None for _ in range(channel.n_clients)]
        self.learning_rate = None

        self.test_mode = False
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


    def __calculate_first_hidden_layer(self, other_id):
        """
        :param other_id:
        :return:
        """

        # 提供数据作为矩阵乘法中的乘数
        def set_triplet_AB():
            self.send_check_msg(self.triplets_id,
                          ComputationMessage(MessageType.SET_TRIPLET, (1, other_id, self.batch_data.shape,
                                                                       self.para.shape)))

        # 提供参数作为矩阵乘法中的的被乘数
        def set_triplet_BA():
            self.send_check_msg(self.triplets_id,
                          ComputationMessage(MessageType.SET_TRIPLET, (2, other_id, self.other_paras[other_id].shape,
                                                                       (self.batch_data.shape[0], self.other_paras[other_id].shape[0]))))

        def get_triples():
            msg = self.receive_check_msg(self.triplets_id, MessageType.TRIPLE_ARRAY)
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
        for client in self.other_data_clients:
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
        updates = self.receive_check_msg(self.server_id, MessageType.CLIENT_OUT_GRAD).data
        own_para_grad = self.batch_data.transpose() @ updates
        portion = np.random.uniform(0, 1, len(self.other_data_clients) + 1)
        portion /= np.sum(portion)
        self.para -= self.learning_rate * own_para_grad * portion[0]
        send_update_threads = []
        for i, data_client in enumerate(self.other_data_clients):
            send_update_threads.append(threading.Thread(
                target=self.__send_updates_to, args=(data_client, own_para_grad * portion[i + 1])
            ))
            send_update_threads[-1].start()

        recv_update_threads = []
        for data_client in self.other_data_clients:
            recv_update_threads.append(threading.Thread(
                target=self.__recv_updates_from, args=(data_client,)
            ))
            recv_update_threads[-1].start()
        for thread in send_update_threads + recv_update_threads:
            thread.join()

    def __train_one_round(self):
        try:
            start_msg = self.receive_check_msg(self.server_id,
                                               [MessageType.NEXT_TRAIN_ROUND, MessageType.TRAINING_STOP])
            if start_msg.header == MessageType.TRAINING_STOP:
                self.logger.log("Received server's training stop message, stop training")
                return False
            else:
                self.test_mode = False
                if start_msg.data == "Test":
                    self.test_mode = True

        except:
            self.logger.logE("Error encountered while receiving server's start message")
            return False

        if not self.test_mode:
            self.batch_data = self.data_loader.get_batch(self.batch_size)
        else:
            self.batch_data = self.test_data_loader.get_batch(self.test_batch_size)

        self.__calc_out_share()
        if self.error:
            self.logger.logE("Error encountered while calculating shared outputs")
            return False

        try:
            self.send_check_msg(self.server_id,
                                ComputationMessage(MessageType.MUL_OUT_SHARE,
                                                   (self.own_out, self.shared_out_AB, self.shared_out_BA)))
        except:
            self.logger.logE("Error encountered while sending output shares to server")
            return False

        if not self.test_mode:
            try:
                self.__parameter_update()
            except:
                self.logger.logE("Error encountered while updateing parameters")
                return False
            if self.error:
                self.logger.logE("Error encountered while updateing parameters")
                return False

        return True

    def start_train(self, wait_for_server: float=100):
        # Get initial training configurations
        self.logger.log("Client started, waiting for server config message with time out %.2f" % wait_for_server)
        try:
            msg = self.receive_check_msg(self.server_id, MessageType.TRAIN_CONFIG, time_out=wait_for_server)
            client_dims = msg.data["client_dims"]
            out_dim = msg.data["out_dim"]
            self.batch_size = msg.data["batch_size"]
            self.test_batch_size = msg.data.get("test_batch_size")
            self.learning_rate = msg.data["learning_rate"]
            self.data_loader.sync_data(msg.data["sync_info"])
            self.test_data_loader.sync_data(msg.data["sync_info"])
            for other_id in client_dims:
                self.other_paras[other_id] = \
                    np.random.normal(0,
                                     1/(len(self.other_data_clients) * client_dims[other_id]),
                                     [client_dims[other_id], out_dim])
            self.para = self.other_paras[self.client_id]
            self.send_check_msg(self.server_id, ComputationMessage(MessageType.CLIENT_READY, None))
        except ClientException:
            self.logger.logE("Train not started")
            return
        except Exception as e:
            self.logger.logE("Python Exception encountered, stop.")
            self.logger.logE("Train not started")
            return

        self.logger.log("Received train conifg message: %s" % msg.data)

        n_rounds = 0
        while True:
            train_res = self.__train_one_round()
            n_rounds += 1
            self.logger.log("Train round %d finished" % n_rounds)
            try:
                self.send_check_msg(self.server_id, ComputationMessage(MessageType.CLIENT_ROUND_OVER, train_res))
            except:
                self.logger.logE("Error encountered while sending round over message to server")
                break
            if not train_res:
                self.logger.logE("Error encountered while training one round. Stop.")
                break


class LabelClient(BaseClient):
    def __init__(self, channel: BaseChannel, label_loader: DataLoader, test_label_loader: DataLoader,
                 server_id: int, loss_func=None, metric_func=None, logger:Logger=None):
        super(LabelClient, self).__init__(channel, logger)
        self.label_loader = label_loader
        self.test_label_loader = test_label_loader
        self.batch_size = None
        self.server_id = server_id
        if loss_func is None:
            self.loss_func = MSELoss()
        else:
            self.loss_func = loss_func
        if metric_func is None:
            self.metric_func = onehot_accuracy
        else:
            self.metric_func = metric_func

        self.test_mode = False
        self.error = False

        # cached labels
        self.batch_labels = None
        self.compute_grad_thread = None

    def __compute_pred_grad(self):
        preds = self.receive_check_msg(self.server_id, MessageType.PRED_LABEL)
        loss = self.loss_func.forward(self.batch_labels, preds.data)

        self.logger.log("Current batch loss: %.4f, accuracy: %.4f" % (loss, self.metric_func(self.batch_labels, preds)))
        grad = self.loss_func.backward()
        self.send_check_msg(self.server_id, ComputationMessage(MessageType.PRED_GRAD, (grad, loss)))

    def __train_one_round(self):
        try:
            start_msg = self.receive_check_msg(self.server_id,
                                               [MessageType.NEXT_TRAIN_ROUND, MessageType.TRAINING_STOP])
            if start_msg.header == MessageType.TRAINING_STOP:
                self.logger.log("Received server's training stop message, stop training")
                return False
            else:
                self.test_mode = False
                if start_msg.data == "Test":
                    self.test_mode = True
        except:
            self.logger.logE("Error encountered while receiving server's start message")
            return False
        try:
            if not self.test_mode:
                self.batch_labels = self.label_loader.get_batch(self.batch_size)
            else:
                self.batch_labels = self.test_label_loader.get_batch(self.test_batch_size)
        except:
            self.logger.logE("Error encountered while loading batch labels")
            return False

        try:
            self.__compute_pred_grad()
        except:
            self.logger.logE("Error encountered while computing prediction gradients")
            return False

        return True

    def start_train(self, wait_for_server:float=100):
        # Get initial training configurations
        self.logger.log("Client started, waiting for server config message with time out %.2f" % wait_for_server)
        try:
            msg = self.receive_check_msg(self.server_id, MessageType.TRAIN_CONFIG, time_out=wait_for_server)
            self.batch_size = msg.data["batch_size"]
            self.test_batch_size = msg.data.get("test_batch_size")
            self.label_loader.sync_data(msg.data["sync_info"])
            self.test_label_loader.sync_data(msg.data["sync_info"])
            self.send_check_msg(self.server_id, ComputationMessage(MessageType.CLIENT_READY, None))
        except ClientException:
            self.logger.logE("Train not started")
            return
        except Exception as e:
            self.logger.logE("Python Exception encountered, stop\n")
            self.logger.logE("Train not started")
            return

        self.logger.log("Received train conifg message: %s" % msg.data)

        n_rounds = 0
        while True:
            train_res = self.__train_one_round()
            n_rounds += 1
            self.logger.log("Train round %d finished" % n_rounds)
            try:
                self.send_check_msg(self.server_id, ComputationMessage(MessageType.CLIENT_ROUND_OVER, train_res))
            except:
                self.logger.logE("Error encountered while sending round over message to server")
                break
            if not train_res:
                self.logger.logE("Error encountered while training one round. Stop.")
                break
