import threading
import tensorflow as tf
import numpy as np
import traceback
from Communication.Message import ComputationMessage, MessageType
from Communication.Channel import BaseChannel
from Client.Client import BaseClient, ClientException
from Utils.Log import Logger

k = tf.keras


class MainTFClient(BaseClient):
    def __init__(self, channel: BaseChannel, data_clients: list, label_client, logger: Logger = None):
        super(MainTFClient, self).__init__(channel, logger)
        self.data_clients = data_clients
        self.label_client = label_client
        self.error = False
        #
        self.data_client_outs = dict()
        #
        self.input_tensor = None
        self.network = None
        self.optimizer = None
        self.network_out = None
        self.tape = None

    def build_network(self, network: k.Model, optimizer: k.optimizers):
        self.network = network
        self.optimizer = optimizer

    def build_default_network(self, input_dim: int, output_dim: int):
        self.network = k.Sequential([
            k.layers.Activation(activation=k.activations.sigmoid, input_shape=(None, input_dim)),
            k.layers.Dense(output_dim, activation=k.activations.sigmoid)])
        self.optimizer = k.optimizers.SGD()

    def __send_config_to(self, client_id: int, config: dict):
        try:
            self.send_check_msg(client_id, ComputationMessage(MessageType.TRAIN_CONFIG, config))
        except Exception as e:
            self.logger.logE("Sending configuration to client %d failed." % client_id)
            self.error = True

    def __receive_client_ready_from(self, client_id: int):
        try:
            self.receive_check_msg(client_id, MessageType.CLIENT_READY)
        except:
            self.logger.logE("Error encountered while receiving client ready message fom client %d" % client_id)
            self.error = True

    def send_config_message(self, config: dict):
        sending_threads = []
        for data_client in self.data_clients + [self.label_client]:
            sending_threads.append(threading.Thread(
                target=self.__send_config_to, args=(data_client, config)
            ))
            sending_threads[-1].start()

        for thread in sending_threads:
            thread.join()

        receiving_threads = []
        for data_client in self.data_clients + [self.label_client]:
            receiving_threads.append(threading.Thread(
                target=self.__receive_client_ready_from, args=(data_client,)
            ))
            receiving_threads[-1].start()

        for thread in receiving_threads:
            thread.join()

        if self.error:
            return False

    def __send_start_message_to(self, client_id, stop=False):
        try:
            if not stop:
                header = MessageType.NEXT_TRAIN_ROUND
            else:
                header = MessageType.TRAINING_STOP
            self.send_check_msg(client_id, ComputationMessage(header, None))
        except:
            self.error = True

    def __broadcast_start(self, stop=False):
        sending_threads = []
        for data_client in self.data_clients + [self.label_client]:
            sending_threads.append(threading.Thread(target=self.__send_start_message_to, args=(data_client, stop)))
            sending_threads[-1].start()
        for thread in sending_threads:
            thread.join()

    def __recv_client_out_from(self, client_id):
        try:
            client_share = self.receive_check_msg(client_id, MessageType.MUL_OUT_SHARE)
            self.data_client_outs[client_id] = client_share.data
        except:
            self.logger.logE("Error encountered while receiving client out from client %d" % client_id)
            self.error = True

    def __gather_client_outs(self):
        gathering_threads = []
        for data_client in self.data_clients:
            gathering_threads.append(threading.Thread(target=self.__recv_client_out_from, args=(data_client,)))
            gathering_threads[-1].start()

        for thread in gathering_threads:
            thread.join()
        if self.error:
            self.logger.logE("Exception encountered while receiving client outputs")
            return None

        output_parts = []
        for data_client in self.data_clients:
            output_part = self.data_client_outs[data_client][0]
            for other_client in self.data_clients:
                if other_client != data_client:
                    output_part += self.data_client_outs[data_client][1][other_client] + \
                                   self.data_client_outs[other_client][2][data_client]
            output_parts.append(output_part)
        return sum(output_parts)

    def __calculate_output(self, input_np: np.ndarray):
        self.input_tensor = tf.Variable(input_np, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            self.network_out = self.network(self.input_tensor)
        self.tape = tape

    def __get_output_grad(self):
        self.send_check_msg(self.label_client, ComputationMessage(MessageType.PRED_LABEL, self.network_out.numpy()))
        grad_server_out = self.receive_check_msg(self.label_client, MessageType.PRED_GRAD)
        return grad_server_out.data[0]

    def __calculate_grad(self, grad_on_output):
        model_jacobians = self.tape.jacobian(self.network_out, self.network.trainable_variables)
        model_grad = [tf.reduce_sum(model_jacobian * (tf.reshape(grad_on_output.astype(np.float32),
                                                                list(grad_on_output.shape) + [1 for i in range(len(model_jacobian.shape) - 2)]) +\
                                                      tf.zeros_like(model_jacobian, dtype=model_jacobian.dtype)),
                                    axis=[0, 1]) for model_jacobian in model_jacobians]
        self.optimizer.apply_gradients(zip(model_grad, self.network.trainable_variables))
        input_jacobian = self.tape.jacobian(self.network_out, self.input_tensor)
        input_grad = tf.reduce_sum(input_jacobian * (tf.reshape(grad_on_output.astype(np.float32),
                                                               list(grad_on_output.shape) + [1 for i in range(len(input_jacobian.shape) - 2)]) +
                                   tf.zeros_like(self.input_tensor, dtype=self.input_tensor.dtype)),
                                   axis=[0, 1]).numpy()
        return input_grad

    def __send_grad_to(self, client_id: int, input_grad: np.ndarray):
        try:
            self.send_check_msg(client_id, ComputationMessage(MessageType.CLIENT_OUT_GRAD, input_grad))
        except:
            self.error = True

    def __send_grads(self, input_grad):
        sending_threads = []
        for i, data_client in enumerate(self.data_clients):
            sending_threads.append(
                threading.Thread(target=self.__send_grad_to, args=(data_client, input_grad)))
            sending_threads[-1].start()
        for thread in sending_threads:
            thread.join()

    def __receive_round_over_from(self, client_id):
        try:
            res = self.receive_check_msg(client_id, MessageType.CLIENT_ROUND_OVER)
            if not res.data:
                self.logger.logE("Received training error data from client %d" % client_id)
                self.error = True
        except:
            self.logger.logE("Error encountered while receiving round over message from client %d" % client_id)
            self.error = True

    def __receive_round_over_msgs(self):
        receiving_threads = []
        for data_client in self.data_clients + [self.label_client]:
            receiving_threads.append(threading.Thread(target=self.__receive_round_over_from, args=(data_client,)))
            receiving_threads[-1].start()
        for thread in receiving_threads:
            thread.join()

    def __train_one_batch(self):
        self.__broadcast_start()
        if self.error:
            self.logger.logE("Error encountered while broadcasting start messages")
            return False

        client_outputs = self.__gather_client_outs()
        if self.error:
            self.logger.logE("Error encountered while gathering client outputs")
            return False

        self.__calculate_output(client_outputs)
        if self.error:
            self.logger.logE("Error encountered while calculating server output")
            return False

        try:
            grad_server_out = self.__get_output_grad()
        except:
            self.logger.logE("Error encountered while getting server output gradient")
            return False

        try:
            input_grad = self.__calculate_grad(grad_server_out)
        except:
            self.logger.logE("Python Error encountered while calculating gradient:\n" + traceback.format_exc())
            return False

        self.__send_grads(input_grad)
        if self.error:
            self.logger.logE("Error encountered while sending grads to clients")
            return False

        self.__receive_round_over_msgs()
        if self.error:
            self.logger.logE("Error encountered while receiving client round over messages")
            return False

        return True

    def start_train(self):
        n_rounds = 0
        while True:
            train_res = self.__train_one_batch()
            n_rounds += 1
            self.logger.log("Train round %d finished" % n_rounds)
            self.logger.log("One train round finished")
            if not train_res:
                self.logger.logE("Training stopped due to error")
                self.__broadcast_start(stop=True)
                break
